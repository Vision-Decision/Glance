# Changelog

## 0.1.0 (2026-11-18)

- First public release: Glance-4B-preview weights (v1, step 3,844) on Hugging Face and the minimal inference package.
- Primitives: `box` (presence gate, patch choice, ordinal edges, IoU head), `choice`, `yesno`, `score`; `ask` answers
  many typed questions in one shared-prefix pass.
- CUDA (bf16, optional `flash-linear-attention`), Apple MPS and CPU (fp32, PyTorch GDN fallback).
