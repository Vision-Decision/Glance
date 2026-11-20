"""Multi-instance box readout (Glance-4B-v1.2b, -v1.3b): 0..N boxes from the same ``<|box_start|>`` readout.

Inference-only port of the training code. The single-box heads are unchanged and still give the single-box answer;
this module adds, on the same features:

  heat   per-patch logit that the patch holds the centre of a referent
  peaks  local maxima of the heat logits on the patch grid (8-neighbourhood), top-k candidates
  boxes  the single-box edge head read at every peak patch (it is patch-conditioned, so a peak on instance i decodes
         instance i's box)
  match  P(candidate is a distinct correct box | present); per-box confidence = P(present) * P(match)
  count  P(n = 1, 2, 3, >= 4 | present); P(n = 0) is 1 - P(present)

Decoding ("count" mode, the default of both released multi-box models): if P(present) < 0.5, no boxes. Otherwise
sort candidates by P(match), drop duplicates by NMS (IoU 0.7), keep the top n with n = argmax of the count head; for
the ">= 4" bucket keep every candidate with P(match) >= ``match_thr`` (at least 4). "thr" mode keeps every candidate
with P(match) >= ``match_thr``. If nothing is kept, the single-box answer is returned.
"""
import math

import torch
import torch.nn as nn

from .heads import _mlp, fourier

_NEI = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class MultiBoxHeads(nn.Module):
    def __init__(self, D, head_dim=512, count_classes=4):
        super().__init__()
        d = head_dim
        self.h_q = nn.Linear(D, d)
        self.h_k = nn.Linear(D, d)
        self.h_scale = nn.Parameter(torch.tensor(1.0))
        self.h_off = nn.Linear(D, 1)
        self.match = _mlp(2 * D + 64 + 1, d, 1)
        self.count = _mlp(D + 1, d, count_classes)

    def heat_logits(self, hq, vis, q_img, valid):
        """hq [M,D], vis [B,N,D] (both layer-normed), valid [M,N] -> [M,N]."""
        dot = torch.einsum("md,mnd->mn", self.h_q(hq), self.h_k(vis)[q_img]) / math.sqrt(self.h_q.out_features)
        dot = dot.masked_fill(~valid, -1e4)
        mx = dot.max(-1, keepdim=True).values
        s = self.h_scale * (dot - mx) + self.h_off(hq) / math.sqrt(hq.shape[-1])
        return (30 * torch.tanh(s / 30)).masked_fill(~valid, -30)


def local_peaks(s, grid_hw, valid, k):
    """Local maxima of s [M,N] on each question's (gh, gw) patch grid, sorted by logit. -> idx [M,k'], ok [M,k']."""
    M, N = s.shape
    gh, gw = grid_hw[:, 0:1].long(), grid_hw[:, 1:2].long().clamp(min=1)
    j = torch.arange(N, device=s.device)[None].expand(M, N)
    r, c = j // gw, j % gw
    peak = valid.clone()
    for dr, dc in _NEI:
        rr, cc = r + dr, c + dc
        inb = (rr >= 0) & (rr < gh) & (cc >= 0) & (cc < gw)
        nj = (rr * gw + cc).clamp(0, N - 1)
        ns = s.gather(1, nj)
        peak &= ~(inb & ((ns > s) | ((ns == s) & (nj < j))))
    v, idx = s.masked_fill(~peak, float("-inf")).topk(min(k, N), -1)
    return idx, torch.isfinite(v)


def multi_forward(box, mb, hq, vis, vis_valid, centers, q_img, grid_hw, k):
    """Candidates for every box question. ``box`` is the single-box :class:`~glance.heads.BoxHeads` (its edge and
    IoU heads decode each candidate); hq / vis are the raw readouts and visual states (normed here, in fp32)."""
    hq = box.q_norm(hq.float())
    vis = box.v_norm(vis.float())
    valid = vis_valid[q_img]
    heat = mb.heat_logits(hq, vis, q_img, valid)
    cidx, cok = local_peaks(heat, grid_hw, valid, k)
    M, K = cidx.shape
    v_k, pc_k = vis[q_img[:, None], cidx], centers[q_img[:, None], cidx]
    hq_k = hq[:, None].expand(M, K, hq.shape[-1])
    feats = torch.cat([hq_k.reshape(M * K, -1), v_k.reshape(M * K, -1), fourier(pc_k.reshape(M * K, 2))], -1)
    el = box.edge(feats).view(M * K, 4, box.K)
    e = (el.softmax(-1) * box.bin_centers).sum(-1)
    pc = pc_k.reshape(M * K, 2)
    cbox = torch.stack([pc[:, 0] - e[:, 0], pc[:, 1] - e[:, 1], pc[:, 0] + e[:, 2], pc[:, 1] + e[:, 3]], -1)
    s_k = heat.gather(1, cidx)
    match = mb.match(torch.cat([hq_k, v_k, fourier(pc_k), s_k[..., None]], -1)).squeeze(-1)
    count = mb.count(torch.cat([hq, torch.log1p(heat.sigmoid().sum(-1, keepdim=True))], -1))
    return dict(cand_box=cbox.view(M, K, 4), cand_ok=cok, cand_match_logit=match, count_logits=count)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _nms(boxes, thr):
    """Greedy NMS over boxes already sorted by score -> kept positions."""
    keep = []
    for i, b in enumerate(boxes):
        if all(_iou(b, boxes[j]) < thr for j in keep):
            keep.append(i)
    return keep


def decode(p_present, single_box, single_iou_p, cands, count_p, mode="count", match_thr=0.5, nms=0.7):
    """One question. cands: list of (box, p_match) for valid candidates; count_p: P(n=1..C | present).
    -> (boxes, confidences), confidences = P(present) * P(match)."""
    if p_present < 0.5:
        return [], []
    if not cands:
        return [single_box], [p_present * single_iou_p]
    if mode == "count":
        c = sorted(cands, key=lambda x: -x[1])
        c = [c[i] for i in _nms([x[0] for x in c], nms)]
        n = int(max(range(len(count_p)), key=lambda i: count_p[i])) + 1
        if n >= len(count_p):
            sel = [x for x in c if x[1] >= match_thr]
            c = sel if len(sel) >= n else c[:n]
        else:
            c = c[:n]
    else:
        c = sorted([x for x in cands if x[1] >= match_thr], key=lambda x: -x[1])
        if not c:
            return [single_box], [p_present * single_iou_p]
        c = [c[i] for i in _nms([x[0] for x in c], nms)]
    return [x[0] for x in c], [p_present * x[1] for x in c]
