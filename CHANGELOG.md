# Changelog

## Unreleased

- Docs: new benchmark table (light and dark) comparing Glance-4B-preview with Qwen3.5-4B, Qwen3-VL-4B/8B and
  Glance-X-4B, including the rows where Glance is behind; cell-by-cell sources in `docs/benchmark_sources.md`.
  ScreenSpot-Pro in the comparison is now read at native resolution (39.3) to match the base-model run.

## 0.1.0 (2026-11-18)

- First public release: Glance-4B-preview weights (v1, step 3,844) on Hugging Face and the minimal inference package.
- Primitives: `box` (presence gate, patch choice, ordinal edges, IoU head), `choice`, `yesno`, `score`; `ask` answers
  many typed questions in one shared-prefix pass.
- CUDA (bf16, optional `flash-linear-attention`), Apple MPS and CPU (fp32, PyTorch GDN fallback).
