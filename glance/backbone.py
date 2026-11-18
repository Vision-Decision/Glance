"""Shared-prefix forward over a Qwen3.5-family hybrid LLM (Gated DeltaNet + gated full attention).

The state (image and/or text) is the *prefix*; every question is a *block* that continues it independently:

* Gated-DeltaNet layers: each block starts from the prefix's final recurrent state and the prefix's last
  (k-1) conv inputs -- exactly what a causal continuation would see, but never another block.
* Full-attention layers: each block attends to all prefix keys plus its own tokens (bidirectional inside
  the block when ``attn_mask == "block_bidir"``, causal otherwise).

This equals running every (prefix + block) separately, but the prefix is computed once, so 8 questions cost
about as much as 1. Layer types and dimensions are read from the loaded Hugging Face modules.

The GDN kernel comes from ``flash-linear-attention`` when it is installed (CUDA); otherwise the pure-PyTorch
reference implementation shipped with transformers is used (works on CPU and Apple MPS, slower).
"""
import torch
import torch.nn.functional as F

try:  # optional fast path (CUDA / Triton)
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gdr
except Exception:  # pragma: no cover
    _fla_chunk_gdr = None


def _gdr(q, k, v, g, beta, initial_state=None):
    """Chunked gated delta rule. q,k: [B,T,Hk,Dk]; v: [B,T,Hv,Dv]; g, beta: [B,T,Hv]. Returns (o, final_state)."""
    if _fla_chunk_gdr is not None and q.is_cuda:
        return _fla_chunk_gdr(q, k, v, g=g, beta=beta, initial_state=initial_state, output_final_state=True,
                              use_qk_l2norm_in_kernel=True)
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    return torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=initial_state,
                                        output_final_state=True, use_qk_l2norm_in_kernel=True)


def _gdn_mix(m, x):
    B, T, _ = x.shape
    mixed = m.in_proj_qkv(x).transpose(1, 2)
    z = m.in_proj_z(x).reshape(B, T, -1, m.head_v_dim)
    beta = m.in_proj_b(x).sigmoid()
    g = -m.A_log.float().exp() * F.softplus(m.in_proj_a(x).float() + m.dt_bias)
    return mixed, z, beta, g


def _gdn_core(m, conv_out, z, beta, g, init):
    B, _, T = conv_out.shape
    conv_out = F.silu(conv_out).transpose(1, 2)
    q, k, v = torch.split(conv_out, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
    q = q.reshape(B, T, -1, m.head_k_dim)
    k = k.reshape(B, T, -1, m.head_k_dim)
    v = v.reshape(B, T, -1, m.head_v_dim)
    o, S = _gdr(q, k, v, g, beta, init)
    o = m.norm(o.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(B, T, -1)
    return m.out_proj(o), S


def _causal_dwconv(x, w, bias, left_pad):
    """Depthwise causal conv1d, x [B,C,L], w [C,1,K]. Uses F.conv1d on CUDA; on CPU / MPS an explicit K-tap sum,
    because grouped conv1d with C=8192 groups is pathologically slow on CPU (~1 s per layer)."""
    K = w.shape[-1]
    if left_pad:
        if x.is_cuda:
            return F.conv1d(x, w, bias, padding=K - 1, groups=x.shape[1])[..., :x.shape[-1]]
        x = F.pad(x, (K - 1, 0))
    elif x.is_cuda:
        return F.conv1d(x, w, bias, padding=0, groups=x.shape[1])
    L = x.shape[-1] - (K - 1)
    out = x[..., 0:L] * w[None, :, 0, 0:1]
    for k in range(1, K):
        out = out + x[..., k:k + L] * w[None, :, 0, k:k + 1]
    return out if bias is None else out + bias[None, :, None]


def gdn_forward(m, xp, xq, p_valid, q_img):
    K = m.conv_kernel_size
    w = m.conv1d.weight  # [C,1,K]
    xp = xp * p_valid[..., None].to(xp.dtype)  # left pads leave the state untouched
    mixed, z, beta, g = _gdn_mix(m, xp)
    Lp = mixed.shape[-1]
    conv = _causal_dwconv(mixed.to(w.dtype), w, m.conv1d.bias, left_pad=True)
    op, S = _gdn_core(m, conv, z, beta, g, None)
    tail = mixed[:, :, Lp - (K - 1):]  # last k-1 conv inputs of each prefix
    mixed_q, zq, beta_q, g_q = _gdn_mix(m, xq)
    mq = torch.cat([tail[q_img], mixed_q], dim=-1).to(w.dtype)
    conv_q = _causal_dwconv(mq, w, m.conv1d.bias, left_pad=False)  # tail gives exactly Tq outputs
    oq, _ = _gdn_core(m, conv_q, zq, beta_q, g_q, S[q_img].contiguous())
    return op, oq


def _qkv(m, x, cos, sin):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    B, T, _ = x.shape
    q, gate = torch.chunk(m.q_proj(x).view(B, T, -1, m.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(B, T, -1)
    q = m.q_norm(q).transpose(1, 2)
    k = m.k_norm(m.k_proj(x).view(B, T, -1, m.head_dim)).transpose(1, 2)
    v = m.v_proj(x).view(B, T, -1, m.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v, gate


def _sdpa(q, k, v, mask, scale):
    if q.shape[1] != k.shape[1]:  # grouped-query attention: expand K/V heads (portable on CPU / MPS)
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


def attn_forward(m, xp, xq, rope_p, rope_q, mask_p, mask_q, q_img):
    B, Lp, _ = xp.shape
    M, Tq, _ = xq.shape
    qp, kp, vp, gp = _qkv(m, xp, *rope_p)
    op = _sdpa(qp, kp, vp, mask_p, m.scaling)
    op = op.transpose(1, 2).reshape(B, Lp, -1) * torch.sigmoid(gp)
    qq, kq, vq, gq = _qkv(m, xq, *rope_q)
    k_all = torch.cat([kp[q_img], kq], dim=2)
    v_all = torch.cat([vp[q_img], vq], dim=2)
    oq = _sdpa(qq, k_all, v_all, mask_q, m.scaling)
    oq = oq.transpose(1, 2).reshape(M, Tq, -1) * torch.sigmoid(gq)
    return m.o_proj(op), m.o_proj(oq)


def build_masks(p_valid, q_valid, q_img, mode):
    """Boolean SDPA masks (True = may attend)."""
    B, Lp = p_valid.shape
    M, Tq = q_valid.shape
    dev = p_valid.device
    causal = torch.ones(Lp, Lp, dtype=torch.bool, device=dev).tril()
    eye_p = torch.eye(Lp, dtype=torch.bool, device=dev)
    mask_p = (causal[None] & p_valid[:, None, :]) | eye_p[None]  # pads attend to themselves (no NaN)
    if mode == "block_bidir":
        within = q_valid[:, None, :].expand(M, Tq, Tq)
    else:
        within = torch.ones(Tq, Tq, dtype=torch.bool, device=dev).tril()[None] & q_valid[:, None, :]
    within = within | torch.eye(Tq, dtype=torch.bool, device=dev)[None]
    to_prefix = p_valid[q_img][:, None, :].expand(M, Tq, Lp)
    mask_q = torch.cat([to_prefix, within], dim=-1)
    return mask_p[:, None], mask_q[:, None]


class SharedPrefixLM(torch.nn.Module):
    """Wraps a loaded ``Qwen3_5ForConditionalGeneration`` (LoRA-injected)."""

    def __init__(self, hf_model, attn_mask="block_bidir"):
        super().__init__()
        self.hf = hf_model
        self.attn_mask = attn_mask
        self.lm = hf_model.model.language_model
        self.layer_types = list(hf_model.config.text_config.layer_types)
        self.hidden = hf_model.config.text_config.hidden_size

    @torch.no_grad()
    def image_features(self, pixel_values, grid_thw):
        out = self.hf.model.get_image_features(pixel_values, grid_thw, return_dict=True)
        return list(out.pooler_output)

    def forward(self, batch, img_feats):
        """Returns final-normed prefix hidden [B,Lp,D] and question-block hidden [M,Tq,D]."""
        emb = self.lm.embed_tokens
        xp = emb(batch["p_ids"])
        if img_feats:
            xp = xp.masked_scatter(batch["p_img_mask"][..., None], torch.cat(img_feats, 0).to(xp.dtype))
        xq = emb(batch["q_ids"])
        p_valid, q_valid, q_img = batch["p_valid"], batch["q_valid"], batch["q_img"]
        rope_p = self.lm.rotary_emb(xp, batch["p_pos"])
        rope_q = self.lm.rotary_emb(xq, batch["q_pos"])
        mask_p, mask_q = build_masks(p_valid, q_valid, q_img, self.attn_mask)
        for layer, ltype in zip(self.lm.layers, self.layer_types):
            hp, hq = layer.input_layernorm(xp), layer.input_layernorm(xq)
            if ltype == "linear_attention":
                ap, aq = gdn_forward(layer.linear_attn, hp, hq, p_valid, q_img)
            else:
                ap, aq = attn_forward(layer.self_attn, hp, hq, rope_p, rope_q, mask_p, mask_q, q_img)
            xp, xq = xp + ap, xq + aq
            xp = xp + layer.mlp(layer.post_attention_layernorm(xp))
            xq = xq + layer.mlp(layer.post_attention_layernorm(xq))
        return self.lm.norm(xp), self.lm.norm(xq)
