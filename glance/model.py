"""High-level API: ``Glance.from_pretrained(...)`` and the typed question helpers."""
import contextlib
import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from .backbone import SharedPrefixLM
from .heads import GlanceHeads
from .prompt import ChatTok, collate, make_item

_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")


# ------------------------------------------------------------------------------------------- questions
@dataclass
class Box:
    """Referring expression -> presence + box."""
    text: str


@dataclass
class Boxes:
    """Referring expression -> presence + every matching box (multi-target). Needs a model with multi-box heads
    (Glance-4B-v1.2b, -v1.3b); on other models it returns the single box when the target is present."""
    text: str


@dataclass
class Choice:
    """Pick one of N options. ``options`` is a list, or a dict label -> description."""
    question: str
    options: Union[Sequence[str], Dict[str, str]]


@dataclass
class YesNo:
    question: str


@dataclass
class Score:
    """Ordinal rating; ``levels`` are ordered lowest first."""
    question: str
    levels: Sequence[str]


Question = Union[str, Box, Boxes, Choice, YesNo, Score, dict]


# ------------------------------------------------------------------------------------------- results
def _r(x, n=4):
    return round(float(x), n)


@dataclass
class BoxResult:
    text: str
    present_prob: float                 # P(target is in the image), calibrated
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels of the input image
    iou_prob: float                     # P(IoU(box, truth) >= 0.5 | present)
    confidence: float                   # present_prob * iou_prob
    present: bool                       # present_prob >= 0.5
    kind: str = "box"

    def __repr__(self):
        b = ", ".join(f"{v:.0f}" for v in self.box)
        return (f"BoxResult({self.text!r}, present_prob={self.present_prob:.3f}, box=[{b}], "
                f"confidence={self.confidence:.3f})")


@dataclass
class BoxesResult:
    text: str
    present_prob: float                 # P(at least one target is in the image), calibrated
    boxes: List[Tuple[float, float, float, float]]  # x1, y1, x2, y2 in pixels; empty if present_prob < 0.5
    confidences: List[float]            # per box: present_prob * P(box is a distinct correct instance)
    count_probs: Dict[str, float]       # P(n = 1, 2, 3, 4+ | present); empty without multi-box heads
    present: bool
    kind: str = "boxes"

    def __repr__(self):
        bs = ", ".join("[" + ", ".join(f"{v:.0f}" for v in b) + "]" for b in self.boxes)
        return f"BoxesResult({self.text!r}, present_prob={self.present_prob:.3f}, boxes=[{bs}])"


@dataclass
class ChoiceResult:
    question: str
    probs: Dict[str, float]
    answer: str
    kind: str = "choice"

    def __repr__(self):
        p = ", ".join(f"{k!r}: {v:.3f}" for k, v in self.probs.items())
        return f"ChoiceResult(answer={self.answer!r}, probs={{{p}}})"


@dataclass
class YesNoResult:
    question: str
    p_yes: float
    answer: bool
    kind: str = "yesno"

    def __repr__(self):
        return f"YesNoResult(p_yes={self.p_yes:.3f})"


@dataclass
class ScoreResult:
    question: str
    probs: Dict[str, float]
    answer: str                         # most likely level
    expected: float                     # expected level index (0 = lowest)
    kind: str = "score"

    def __repr__(self):
        p = ", ".join(f"{k!r}: {v:.3f}" for k, v in self.probs.items())
        return f"ScoreResult(answer={self.answer!r}, expected={self.expected:.2f}, probs={{{p}}})"


def to_dict(r):
    return asdict(r)


# ------------------------------------------------------------------------------------------- model
def _auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _norm_question(q):
    if isinstance(q, (Box, Boxes, Choice, YesNo, Score)):
        return q
    if isinstance(q, str):
        return Box(q)
    if isinstance(q, dict):
        t = q.get("type", "box")
        if t == "box":
            return Box(q["text"])
        if t == "boxes":
            return Boxes(q["text"])
        if t == "choice":
            return Choice(q["question"], q["options"])
        if t in ("yesno", "noul"):
            return YesNo(q["question"])
        if t == "score":
            return Score(q["question"], q.get("levels") or q["options"])
    raise TypeError(f"unsupported question: {q!r}")


class Glance:
    """Glance typed-decision model.

    >>> m = Glance.from_pretrained("Vision-Decision/Glance-4B-preview")
    >>> m.box("photo.jpg", "the red mug on the left")
    """

    def __init__(self, net, heads, processor, cfg, device, dtype, autocast, multibox=None):
        self.net, self.heads, self.processor, self.cfg = net, heads, processor, cfg
        self.multibox = multibox  # MultiBoxHeads or None
        self.device, self.dtype, self._autocast = device, dtype, autocast
        self.tok = ChatTok(processor.tokenizer, cfg["max_state_tokens"], cfg["max_q_tokens"], cfg["max_q_tokens_typed"])
        self.temperatures = dict(cfg.get("temperatures", {}))

    # ----------------------------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(cls, repo_or_path="Vision-Decision/Glance-4B-preview", device=None, dtype=None, base_model=None,
                        revision=None):
        """Load the base model from the Hub, inject the LoRA adapter and load the heads.

        device: "cuda" | "mps" | "cpu" | None (auto). dtype: None picks bfloat16 on CUDA (LoRA kept in fp32
        under autocast, as in training) and float32 elsewhere.
        """
        from safetensors.torch import load_file
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

        if os.path.isdir(repo_or_path):
            path = lambda f: os.path.join(repo_or_path, f)
        else:
            from huggingface_hub import hf_hub_download
            path = lambda f: hf_hub_download(repo_or_path, f, revision=revision)
        cfg = json.load(open(path("glance_config.json")))
        device = device or _auto_device()
        if dtype is None:
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        autocast = device.startswith("cuda") and dtype == torch.bfloat16
        base_id = base_model or cfg["base_model"]

        hf = AutoModelForImageTextToText.from_pretrained(base_id, dtype=dtype, attn_implementation="sdpa")
        processor = AutoProcessor.from_pretrained(base_id)
        hf.requires_grad_(False)
        lc = cfg["lora"]
        inject_adapter_in_model(LoraConfig(r=lc["r"], lora_alpha=lc["lora_alpha"], lora_dropout=0.0,
                                           target_modules=lc["target_modules"], bias="none"), hf)
        res = set_peft_model_state_dict(hf, load_file(path("adapter_model.safetensors")))
        bad = [k for k in getattr(res, "unexpected_keys", []) if "lora" in k]
        if bad:
            raise RuntimeError(f"unexpected LoRA keys: {bad[:5]}")
        lora_dtype = torch.float32 if autocast else dtype
        for n, p in hf.named_parameters():
            if "lora_" in n:
                p.data = p.data.to(lora_dtype)
        if autocast:  # let autocast handle bf16 x fp32 matmuls instead of upcasting activations
            for m in hf.modules():
                if hasattr(m, "lora_A"):
                    m.cast_input_dtype_enabled = False
        hf.to(device).eval()

        net = SharedPrefixLM(hf, cfg.get("attn_mask", "block_bidir")).eval()
        hsd = load_file(path("heads.safetensors"))
        E = hsd["typed.E"]
        gi = hsd["typed.group_idx"]
        heads = GlanceHeads(net.hidden, cfg["heads"], (E.shape[0], gi.shape[0], gi.shape[1]))
        heads.load_state_dict(hsd)
        heads.to(device).float().eval()
        multibox = None
        mcfg = cfg.get("multibox")
        if mcfg:
            from .multibox import MultiBoxHeads
            msd = load_file(path(mcfg.get("file", "multibox_heads.safetensors")))
            msd = {k[len("box.multi."):] if k.startswith("box.multi.") else k: v for k, v in msd.items()}
            multibox = MultiBoxHeads(net.hidden, cfg["heads"]["head_dim"], mcfg.get("count_classes", 4))
            multibox.load_state_dict(msd)
            multibox.to(device).float().eval()
        return cls(net, heads, processor, cfg, device, dtype, autocast, multibox)

    # ----------------------------------------------------------------------------------- core
    def _state(self, state, text):
        """-> (PIL image or None, state text or None)."""
        from PIL import Image
        img = None
        if isinstance(state, Image.Image):
            img = state
        elif isinstance(state, (str, os.PathLike)) and str(state).lower().endswith(_IMG_EXT) and os.path.isfile(state):
            img = Image.open(state)
        elif isinstance(state, str):
            if text is not None:
                raise ValueError("state is text and text= was also given")
            text = state
        elif state is not None:
            raise TypeError("state must be a PIL image, an image path, a string, or None")
        if img is not None:
            img = img.convert("RGB")
        if img is None and not text:
            raise ValueError("need an image and/or text state")
        return img, text

    @torch.no_grad()
    def ask(self, state, questions: List[Question], text: Optional[str] = None, max_vis_tokens: Optional[int] = None,
            max_upscale: Optional[float] = None, calibrated: bool = True):
        """Answer many typed questions about one state in a single shared-prefix forward pass.

        state: PIL image, image path, or a text string (or None with ``text=``). ``text=`` adds text after the
        image. questions: ``Box`` / ``Boxes`` / ``Choice`` / ``YesNo`` / ``Score`` objects, dicts, or plain strings
        (= Box).
        max_vis_tokens: visual-token budget (1 token = 32x32 px). 640 matches natural-image training; use
        ~2048 for screenshots. Returns one result object per question, in order.
        """
        img, text = self._state(state, text)
        qs = [_norm_question(q) for q in questions]
        if any(isinstance(q, (Box, Boxes)) for q in qs) and img is None:
            raise ValueError("box questions need an image")
        blocks, labs = [], []
        for q in qs:
            if isinstance(q, (Box, Boxes)):
                blocks.append(self.tok.box_block(q.text)); labs.append(None)
            elif isinstance(q, Choice):
                opts = [f"{k}: {v}" if v else str(k) for k, v in q.options.items()] if isinstance(q.options, dict) \
                    else list(q.options)
                ids, lab = self.tok.typed_block("choice", q.question, opts); blocks.append(ids); labs.append(lab)
            elif isinstance(q, YesNo):
                ids, lab = self.tok.typed_block("yesno", q.question); blocks.append(ids); labs.append(lab)
            else:
                ids, lab = self.tok.typed_block("score", q.question, list(q.levels)); blocks.append(ids); labs.append(lab)
        item = make_item(self.tok, self.processor.image_processor, img, text, blocks,
                         max_vis_tokens or self.cfg["max_vis_tokens"],
                         max_upscale if max_upscale is not None else self.cfg["max_upscale"])
        multi = self.multibox is not None and any(isinstance(q, Boxes) for q in qs)
        out = self._forward(collate([item], self.tok), multi=multi)
        T = self.temperatures if calibrated else {}
        res = []
        for m, (q, lab) in enumerate(zip(qs, labs)):
            if isinstance(q, Box):
                W, H = item["orig_wh"]
                pl = float(out["presence_logit"][m]) / T.get("presence", 1.0)
                pp = 1 / (1 + math.exp(-pl))
                pi = float(torch.sigmoid(out["iou_logit"][m]))
                b = out["box"][m].clamp(0, 1) * torch.tensor([W, H, W, H], dtype=torch.float32)
                res.append(BoxResult(q.text, _r(pp), tuple(round(float(v), 1) for v in b), _r(pi), _r(pp * pi),
                                     pp >= 0.5))
                continue
            if isinstance(q, Boxes):
                res.append(self._boxes_result(q, out, m, item["orig_wh"], T))
                continue
            lg = out["label_logits"][m, lab]
            if isinstance(q, YesNo):
                p = torch.softmax(lg / T.get("noul", 1.0), -1)
                res.append(YesNoResult(q.question, _r(p[1]), bool(p[1] >= 0.5)))
            elif isinstance(q, Choice):
                p = torch.softmax(lg / T.get("choice", 1.0), -1)
                keys = list(q.options.keys()) if isinstance(q.options, dict) else [str(o) for o in q.options]
                probs = {k: _r(v) for k, v in zip(keys, p.tolist())}
                res.append(ChoiceResult(q.question, probs, keys[int(p.argmax())]))
            else:
                p = torch.softmax(lg / T.get("score", 1.0), -1)
                keys = [str(o) for o in q.levels]
                res.append(ScoreResult(q.question, {k: _r(v) for k, v in zip(keys, p.tolist())},
                                       keys[int(p.argmax())], _r((p * torch.arange(len(keys))).sum(), 3)))
        return res

    def _boxes_result(self, q, out, m, wh, T):
        from .multibox import decode
        W, H = wh
        scale = torch.tensor([W, H, W, H], dtype=torch.float32)
        pp = 1 / (1 + math.exp(-float(out["presence_logit"][m]) / T.get("presence", 1.0)))
        single = [float(v) for v in out["box"][m].clamp(0, 1) * scale]
        pi = float(torch.sigmoid(out["iou_logit"][m]))
        cands, count_p = [], []
        if "multi" in out:
            mo = out["multi"]
            ok = mo["cand_ok"][m]
            cb = mo["cand_box"][m][ok].clamp(0, 1) * scale
            mp = torch.sigmoid(mo["cand_match_logit"][m][ok])
            cands = [([float(v) for v in b], float(p)) for b, p in zip(cb, mp)]
            count_p = mo["count_logits"][m].softmax(-1).tolist()
        mc = self.cfg.get("multibox") or {}
        boxes, conf = decode(pp, single, pi, cands, count_p, mode=mc.get("decode", "count"),
                             match_thr=mc.get("match_thr", 0.5), nms=mc.get("nms", 0.7))
        labels = [str(i + 1) for i in range(len(count_p))]
        if labels:
            labels[-1] += "+"
        return BoxesResult(q.text, _r(pp), [tuple(round(v, 1) for v in b) for b in boxes], [_r(c) for c in conf],
                           {k: _r(v) for k, v in zip(labels, count_p)}, pp >= 0.5)

    def _forward(self, batch, multi=False):
        dev = self.device
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        feats = []
        if b["pixel_values"] is not None:
            feats = self.net.image_features(b["pixel_values"].to(self.dtype), b["grid_thw"])
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if self._autocast else contextlib.nullcontext()
        with ctx:
            hp, hq = self.net(b, feats)
        M = hq.shape[0]
        h = hq[torch.arange(M, device=hq.device), b["q_last"]]
        out = dict(label_logits=self.heads.typed(h).float().cpu())
        if b["pixel_values"] is not None:
            B, N = b["vis_valid"].shape
            vis = hp.new_zeros(B, N, hp.shape[-1])
            vis[b["vis_valid"]] = hp[b["p_img_mask"]]
            o = self.heads.box(h, vis, b["vis_valid"], b["centers"], b["q_img"])
            out.update({k: o[k].float().cpu() for k in ("presence_logit", "iou_logit", "box", "center_logits")})
            if multi:
                from .multibox import multi_forward
                g = b["grid_thw"]
                ghw = torch.stack([g[:, 1] // 2, g[:, 2] // 2], -1).to(hq.device)[b["q_img"]]
                mo = multi_forward(self.heads.box, self.multibox, h, vis, b["vis_valid"], b["centers"], b["q_img"],
                                   ghw, self.cfg["multibox"].get("k", 16))
                out["multi"] = {k: v.float().cpu() if v.is_floating_point() else v.cpu() for k, v in mo.items()}
        return out

    # ----------------------------------------------------------------------------------- one-question helpers
    def box(self, image, text, **kw) -> BoxResult:
        """Locate a referring expression. Returns presence probability, pixel box and box confidence."""
        return self.ask(image, [Box(text)], **kw)[0]

    def boxes(self, image, text, **kw) -> BoxesResult:
        """Locate every instance a referring expression covers (multi-box models; see :class:`Boxes`)."""
        return self.ask(image, [Boxes(text)], **kw)[0]

    def choice(self, state, question, options, **kw) -> ChoiceResult:
        return self.ask(state, [Choice(question, options)], **kw)[0]

    def yesno(self, state, question, **kw) -> YesNoResult:
        return self.ask(state, [YesNo(question)], **kw)[0]

    def score(self, state, question, levels, **kw) -> ScoreResult:
        return self.ask(state, [Score(question, levels)], **kw)[0]
