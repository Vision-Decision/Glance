# Changelog

## 0.2.0 (2026-11-20)

- Earlier checkpoints published as their own Hugging Face repos: Glance-4B-v1.1, -v1.2, -v1.2b, -v1.3 and -v1.3b,
  each with a model card (changes, full results against the pre-registered bars, known regressions). All load with
  `Glance.from_pretrained`.
- Multi-box readout: `Boxes(text)` questions and `Glance.boxes(image, text)` return every instance an expression
  covers on models that ship `multibox_heads.safetensors` (v1.2b, v1.3b). Inference-only port of the training code;
  on single-box models `Boxes` returns the single box when the target is present.
- `examples/quick_bench.py`: a few-minute smoke benchmark on fixed public subsets (RefCOCO val 200, gRefCOCO
  no-target 200 and multi-target 100, ScreenSpot-v2 100, latency for 1 and 8 questions); item ids in
  `examples/quick_bench_items.json`. README: version comparison on the quick bench.

## Unreleased (docs, before 0.2.0)

- Docs: new benchmark table (light and dark) comparing Glance-4B-preview with Qwen3.5-4B, Qwen3-VL-4B/8B and
  Glance-X-4B, including the rows where Glance is behind; cell-by-cell sources in `docs/benchmark_sources.md`.
  ScreenSpot-Pro in the comparison is now read at native resolution (39.3) to match the base-model run.

## 0.1.0 (2026-11-18)

- First public release: Glance-4B-preview weights (v1, step 3,844) on Hugging Face and the minimal inference package.
- Primitives: `box` (presence gate, patch choice, ordinal edges, IoU head), `choice`, `yesno`, `score`; `ask` answers
  many typed questions in one shared-prefix pass.
- CUDA (bf16, optional `flash-linear-attention`), Apple MPS and CPU (fp32, PyTorch GDN fallback).
