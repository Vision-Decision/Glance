"""Readout heads.

Box primitive (geometry in normalized image coordinates, visual token (i, j) of a gh x gw grid has center
((j+.5)/gw, (i+.5)/gh)):
  * presence gate      P(present) from the block readout;
  * center             softmax over the image's visual tokens (one per 32x32 px patch);
  * edges              4 ordinal distributions (K bins over [edge_lo, edge_hi]) of the left/top/right/bottom
                       offsets from the chosen patch center; the box is their expectation;
  * IoU head           P(IoU >= 0.5) of the decoded box, so box confidence = P(present) * P(IoU >= 0.5).

Typed primitives (choice / score / yes-no): a softmax over the block's label tokens (A..Z, or {No, Yes}).
Label logits = h @ E[label tokens]^T (frozen LM-head rows) + a trained linear residual.
"""
import math
import torch
import torch.nn as nn


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def fourier(xy, n=16):
    freqs = (2.0 ** torch.arange(n, device=xy.device, dtype=xy.dtype)) * math.pi
    a = xy[..., None] * freqs
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


class BoxHeads(nn.Module):
    def __init__(self, D, head_dim=512, edge_bins=64, edge_lo=-0.1, edge_hi=1.0):
        super().__init__()
        d, K = head_dim, edge_bins
        self.K = K
        self.q_norm = nn.LayerNorm(D)
        self.v_norm = nn.LayerNorm(D)
        self.presence = _mlp(D, d, 1)
        self.c_q = nn.Linear(D, d)
        self.c_k = nn.Linear(D, d)
        self.edge = _mlp(2 * D + 64, 2 * d, 4 * K)
        self.iou = _mlp(2 * D + 64, d, 1)
        self.register_buffer("bin_centers", torch.linspace(edge_lo, edge_hi, K), persistent=False)

    def forward(self, hq, vis, vis_valid, centers, q_img):
        """hq [M,D] block readouts; vis [B,N,D] visual-token hidden states; vis_valid [B,N]; centers [B,N,2];
        q_img [M] which prefix each question belongs to. Runs in fp32."""
        hq = self.q_norm(hq.float())
        vis = self.v_norm(vis.float())
        pres = self.presence(hq).squeeze(-1)
        kv = self.c_k(vis)
        cl = torch.einsum("md,mnd->mn", self.c_q(hq), kv[q_img]) / math.sqrt(self.c_q.out_features)
        cl = cl.masked_fill(~vis_valid[q_img], -1e4)
        idx = cl.argmax(-1)
        v_sel, pc = vis[q_img, idx], centers[q_img, idx]
        feats = torch.cat([hq, v_sel, fourier(pc)], -1)
        el = self.edge(feats).view(-1, 4, self.K)
        e = (el.softmax(-1) * self.bin_centers).sum(-1)  # expected l, t, r, b offsets
        box = torch.stack([pc[:, 0] - e[:, 0], pc[:, 1] - e[:, 1], pc[:, 0] + e[:, 2], pc[:, 1] + e[:, 3]], -1)
        return dict(presence_logit=pres, center_logits=cl, center_idx=idx, edge_logits=el, box=box,
                    iou_logit=self.iou(feats).squeeze(-1))


class TypedHead(nn.Module):
    """Label-token readout. ``E`` and ``group_idx`` are loaded from the checkpoint."""

    def __init__(self, D, n_label_tokens=30, n_groups=28, max_group=2):
        super().__init__()
        self.register_buffer("E", torch.zeros(n_label_tokens, D))
        self.register_buffer("group_idx", torch.full((n_groups, max_group), -1, dtype=torch.long))
        self.res = nn.Linear(D, n_groups)

    def forward(self, h):
        """h [M,D] -> [M,G] label logits (fp32). Groups 0..25 = letters A..Z, 26 = no, 27 = yes."""
        h = h.float()
        tl = h @ self.E.t()
        gi = self.group_idx
        g = tl[:, gi.clamp(min=0)].masked_fill((gi < 0)[None], float("-inf"))
        return torch.logsumexp(g, -1) + self.res(h)


class GlanceHeads(nn.Module):
    def __init__(self, D, head_cfg, typed_shape):
        super().__init__()
        self.box = BoxHeads(D, **head_cfg)
        self.typed = TypedHead(D, *typed_shape)
