# Models and data on Hugging Face

Everything below is public under the [Vision-Decision](https://huggingface.co/Vision-Decision) organisation. The same
items are grouped in four collections:
[Glance models](https://huggingface.co/collections/Vision-Decision/glance-models-6affec89a20ff6e9d31de39f),
[Glance training data](https://huggingface.co/collections/Vision-Decision/glance-training-data-6affec8bf34e26467ed86667),
[Glance evaluation](https://huggingface.co/collections/Vision-Decision/glance-evaluation-6affec8fdeb34f65d8f366c9) and
[Collaborator projects](https://huggingface.co/collections/Vision-Decision/collaborator-projects-6affec914cc1ae6192af1651).
Each repository's card gives its sources, what was removed and why, and its licence.

## Models

| repository | what it is | licence |
|---|---|---|
| [Glance-4B-preview](https://huggingface.co/Vision-Decision/Glance-4B-preview) | Glance v1: LoRA adapter and decision heads on Qwen3.5-4B. This is the model the `glance` package loads by default. | CC BY-NC 4.0 |
| [Glance-4B-v1.1](https://huggingface.co/Vision-Decision/Glance-4B-v1.1) | v1 recipe plus high-resolution GUI data; superseded, kept for reproducibility. | CC BY-NC 4.0 |
| [Glance-4B-v1.2](https://huggingface.co/Vision-Decision/Glance-4B-v1.2) | v1.1 plus expressions without category words and matched negatives; superseded. | CC BY-NC 4.0 |
| [Glance-4B-v1.2b](https://huggingface.co/Vision-Decision/Glance-4B-v1.2b) | v1.2 plus multi-box heads; superseded by v1.3b. | CC BY-NC 4.0 |
| [Glance-4B-v1.3](https://huggingface.co/Vision-Decision/Glance-4B-v1.3) | re-balanced v1.2; recommended for single-box use. | CC BY-NC 4.0 |
| [Glance-4B-v1.3b](https://huggingface.co/Vision-Decision/Glance-4B-v1.3b) | v1.3 plus multi-box heads; recommended for multi-target use (needs `glance` 0.2.0). | CC BY-NC 4.0 |

## Data

| repository | what it is | licence |
|---|---|---|
| [Glance-COCO](https://huggingface.co/datasets/Vision-Decision/Glance-COCO) | Grounding on COCO images in English and Chinese. `default`: a specificity ladder, multi-target and function requests, and hard negatives, with answers from COCO annotations (formerly Glance-Data). `plus`: training-only supplements without the category word, with position words and exact counts on positives. Also the trainer-format records (`records/`), the `robust` evaluation set (paraphrase, function, vague, Chinese) and the build intermediates. | CC BY-NC-SA 4.0 |
| [Glance-GUI](https://huggingface.co/datasets/Vision-Decision/Glance-GUI) | Synthetic GUI grounding (guisyn-v0): 18.5k desktop and web screenshots, 1.08M requests, with hard negatives checked against the accessibility tree. | annotations CC BY 4.0; screenshots research only |
| [Glance-Video](https://huggingface.co/datasets/Vision-Decision/Glance-Video) | Typed-question records for short video clips (Kinetics-700, Countix, RepCount-A). Annotations only, no video. | mixed, see card |
| [Glance-Experiments](https://huggingface.co/datasets/Vision-Decision/Glance-Experiments) | Results and model outputs: Glance-4B-preview on the Decision Index public suites, the referring-negative audit, scan-engine measurements, architecture summaries. | mixed, see card |

On 2026-11-20 the earlier per-release repositories (Glance-Data-Plus-clean, glance-ground-v2, glance-robust-ref-clean,
glance-train-data-clean, guisyn-v0-clean, glance-video-annotations, di-runs-clean, grounding-negatives-clean,
collection-scan-results-clean, collection-scan-data-clean, scan-pilot-annotations) were merged into these four. Every
file moved unchanged; each original card is kept next to its files.

## From collaborators

| repository | what it is | licence |
|---|---|---|
| [Glance-X-4B](https://huggingface.co/Vision-Decision/Glance-X-4B) | Glance-X: one set of 4B weights that can cache the image, the request or both, with multi-box output. | CC BY-NC-SA 4.0 |
| [Glance-X-Lite](https://huggingface.co/Vision-Decision/Glance-X-Lite) | A 0.8B Glance-X with image-first and text-first directions. | CC BY-NC-SA 4.0 |

Images are not redistributed. The COCO-based datasets reference COCO 2014 train images by id and file name; download
them from [cocodataset.org](https://cocodataset.org).
