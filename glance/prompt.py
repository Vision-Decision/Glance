"""Preprocessing: image resize, chat-style prompt layout, and batch construction.

Layout (the state is the shared prefix; each question is a block that continues it):

  prefix : <|im_start|>user\\n [<|vision_start|><|image_pad|>*n<|vision_end|>] [state text]
  box    : \\n<|object_ref_start|>{expr}<|object_ref_end|><|im_end|>\\n<|im_start|>assistant\\n<|box_start|>
  choice : \\n\\n{question}\\n\\nOptions:\\nA. ..\\nB. ..\\nAnswer with the letter of the best option.<|im_end|>
           \\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
  score  : same, with "Scale (lowest to highest):" and "Answer with the letter of the level."
  yes/no : \\n\\n{question}\\nAnswer yes or no.<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n

The readout is the last token of each block.
"""
import math
import torch

LETTERS = [chr(ord("A") + i) for i in range(26)]
NO_GROUP, YES_GROUP = 26, 27


def resize_dims(W, H, max_tokens, max_upscale, factor=32):
    """Largest (W', H'), multiples of ``factor``, aspect ratio kept, with (W'/f) * (H'/f) <= max_tokens."""
    s = min(math.sqrt(max_tokens * factor * factor / (W * H)), max_upscale)
    while True:
        w = max(factor, int(round(W * s / factor)) * factor)
        h = max(factor, int(round(H * s / factor)) * factor)
        if (w // factor) * (h // factor) <= max_tokens:
            return w, h
        s *= 0.98


class ChatTok:
    def __init__(self, tokenizer, max_state_tokens=1536, max_q_tokens=40, max_q_tokens_typed=384):
        self.tok = tokenizer
        t = tokenizer.convert_tokens_to_ids
        self.vs, self.ve, self.pad_img = t("<|vision_start|>"), t("<|vision_end|>"), t("<|image_pad|>")
        self.ors, self.ore, self.bs = t("<|object_ref_start|>"), t("<|object_ref_end|>"), t("<|box_start|>")
        self.ims, self.ime = t("<|im_start|>"), t("<|im_end|>")
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.user_open = [self.ims] + self.enc("user\n")
        self.asst_typed = [self.ime] + self.enc("\n") + [self.ims] + self.enc("assistant\n<think>\n\n</think>\n\n")
        self.asst_box = [self.ime] + self.enc("\n") + [self.ims] + self.enc("assistant\n") + [self.bs]
        self.max_state, self.max_q_box, self.max_q_typed = max_state_tokens, max_q_tokens, max_q_tokens_typed

    def enc(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def box_block(self, text):
        ids = self.enc(text)[: self.max_q_box]
        return self.enc("\n") + [self.ors] + ids + [self.ore] + self.asst_box

    def typed_block(self, kind, question, options=None):
        """-> (token ids, label-group indices)."""
        if kind == "yesno":
            body = "\n\n" + question.strip() + "\nAnswer yes or no."
            labs = [NO_GROUP, YES_GROUP]
        else:
            if not 2 <= len(options) <= 26:
                raise ValueError("choice / score need between 2 and 26 options")
            head = "Options:" if kind == "choice" else "Scale (lowest to highest):"
            tail = "Answer with the letter of the best option." if kind == "choice" else "Answer with the letter of the level."
            body = "\n\n" + question.strip() + f"\n\n{head}\n" + "".join(
                f"{LETTERS[i]}. {str(o).strip()}\n" for i, o in enumerate(options)) + tail
            labs = list(range(len(options)))
        ids = self.enc(body)
        if len(ids) > self.max_q_typed:  # keep the end (options + answer instruction)
            ids = ids[-self.max_q_typed:]
        return ids + self.asst_typed, labs

    def state_ids(self, text):
        return self.enc(text)[: self.max_state]


def make_item(tok, image_processor, image, text, blocks, max_vis_tokens, max_upscale):
    """One state + its question blocks -> item dict for :func:`collate`. ``image`` is a PIL RGB image or None."""
    from PIL import Image
    it = dict(q_ids=blocks)
    segs = [("ids", tok.user_open)]
    if image is not None:
        W, H = image.size
        w, h = resize_dims(W, H, max_vis_tokens, max_upscale)
        img = image.resize((w, h), Image.BICUBIC)
        pv = image_processor(images=[img], do_resize=False, return_tensors="pt")
        it["pixel_values"], it["grid_thw"] = pv["pixel_values"], pv["image_grid_thw"][0]
        it["orig_wh"], it["in_wh"] = (W, H), (w, h)
        segs += [("ids", [tok.vs]), ("img", None), ("ids", [tok.ve])]
    if text:
        segs.append(("ids", tok.state_ids(text)))
    it["segs"] = segs
    return it


def collate(items, tok, merge=2, pad_to=64):
    """Items -> batch tensors. Prefixes are left-padded to a multiple of ``pad_to``; blocks right-padded."""
    B = len(items)
    seq, ghw = [], []
    for it in items:
        if "grid_thw" in it:
            g = it["grid_thw"]
            gh, gw = int(g[1] // merge), int(g[2] // merge)
        else:
            gh = gw = 0
        ghw.append((gh, gw))
        seq.append(sum(len(s[1]) if s[0] == "ids" else gh * gw for s in it["segs"]))
    Lp = -(-max(seq) // pad_to) * pad_to
    Nv = max(1, max(gh * gw for gh, gw in ghw))
    p_ids = torch.full((B, Lp), tok.pad, dtype=torch.long)
    p_valid = torch.zeros(B, Lp, dtype=torch.bool)
    p_img = torch.zeros(B, Lp, dtype=torch.bool)
    p_pos = torch.zeros(3, B, Lp, dtype=torch.long)
    vis_valid = torch.zeros(B, Nv, dtype=torch.bool)
    centers = torch.zeros(B, Nv, 2)
    q_start = []
    for b, it in enumerate(items):
        gh, gw = ghw[b]
        s = Lp - seq[b]
        p_valid[b, s:] = True
        cur = 0
        for kind, ids in it["segs"]:
            if kind == "ids":
                n = len(ids)
                p_ids[b, s:s + n] = torch.tensor(ids, dtype=torch.long)
                p_pos[:, b, s:s + n] = cur + torch.arange(n)
                s += n
                cur += n
            else:  # visual tokens: 3-D (t, h, w) rotary positions
                n = gh * gw
                ii, jj = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
                p_ids[b, s:s + n] = tok.pad_img
                p_img[b, s:s + n] = True
                p_pos[0, b, s:s + n] = cur
                p_pos[1, b, s:s + n] = cur + ii.flatten()
                p_pos[2, b, s:s + n] = cur + jj.flatten()
                vis_valid[b, :n] = True
                centers[b, :n, 0] = (jj.flatten() + 0.5) / gw
                centers[b, :n, 1] = (ii.flatten() + 0.5) / gh
                s += n
                cur += max(gh, gw)
        q_start.append(cur)
    q_list, q_img = [], []
    for b, it in enumerate(items):
        for q in it["q_ids"]:
            q_list.append(q)
            q_img.append(b)
    M = len(q_list)
    Tq = -(-max(len(q) for q in q_list) // 16) * 16
    q_ids = torch.full((M, Tq), tok.pad, dtype=torch.long)
    q_valid = torch.zeros(M, Tq, dtype=torch.bool)
    q_last = torch.zeros(M, dtype=torch.long)
    q_pos = torch.zeros(3, M, Tq, dtype=torch.long)
    for m, (q, b) in enumerate(zip(q_list, q_img)):
        q_ids[m, :len(q)] = torch.tensor(q)
        q_valid[m, :len(q)] = True
        q_last[m] = len(q) - 1
        q_pos[:, m, :] = q_start[b] + torch.arange(Tq)
    pv = [it["pixel_values"] for it in items if "pixel_values" in it]
    gt = [it["grid_thw"] for it in items if "grid_thw" in it]
    return dict(pixel_values=torch.cat(pv) if pv else None, grid_thw=torch.stack(gt) if gt else None,
                p_ids=p_ids, p_valid=p_valid, p_img_mask=p_img, p_pos=p_pos,
                q_ids=q_ids, q_valid=q_valid, q_last=q_last, q_pos=q_pos, q_img=torch.tensor(q_img),
                vis_valid=vis_valid, centers=centers)
