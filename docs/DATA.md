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

## Training data

| repository | what it is | licence |
|---|---|---|
| [Glance-Data](https://huggingface.co/datasets/Vision-Decision/Glance-Data) | Grounding requests on COCO images in English and Chinese: a specificity ladder, multi-target and function requests, and hard negatives. Answers come from COCO annotations. | CC BY-NC-SA 4.0 |
| [Glance-Data-Plus-clean](https://huggingface.co/datasets/Vision-Decision/Glance-Data-Plus-clean) | A training-only supplement to Glance-Data with the same schema: requests without the category word, position words and exact counts on positives, and matching negatives. | CC BY-NC-SA 4.0 |
| [glance-ground-v2](https://huggingface.co/datasets/Vision-Decision/glance-ground-v2) | Glance-Data converted to the record format the Glance trainer reads, plus sole-instance position requests and category-free records. | CC BY-NC-SA 4.0 |
| [guisyn-v0-clean](https://huggingface.co/datasets/Vision-Decision/guisyn-v0-clean) | Synthetic GUI grounding: 18.5k desktop and web screenshots, 1.08M requests, with hard negatives checked against the accessibility tree. | annotations CC BY 4.0; screenshots research only |
| [glance-video-annotations](https://huggingface.co/datasets/Vision-Decision/glance-video-annotations) | Typed-question records for short video clips (Kinetics-700, Countix, RepCount-A). Annotations only, no video. | mixed, see card |

## Evaluation

| repository | what it is | licence |
|---|---|---|
| [glance-robust-ref-clean](https://huggingface.co/datasets/Vision-Decision/glance-robust-ref-clean) | Referring-expression grounding when the request is phrased unusually: paraphrases, function, vague and relational words, and Chinese, each with absent variants. | CC BY-NC-SA 4.0 |
| [di-runs-clean](https://huggingface.co/datasets/Vision-Decision/di-runs-clean) | The complete run of Glance-4B-preview on the Decision Index public suites (text and vision), with request payloads removed. | CC BY 4.0 |

## From collaborators

| repository | what it is | licence |
|---|---|---|
| [Glance-X-4B](https://huggingface.co/Vision-Decision/Glance-X-4B) | Glance-X: one set of 4B weights that can cache the image, the request or both, with multi-box output. | CC BY-NC-SA 4.0 |
| [Glance-X-Lite](https://huggingface.co/Vision-Decision/Glance-X-Lite) | A 0.8B Glance-X with image-first and text-first directions. | CC BY-NC-SA 4.0 |

Images are not redistributed. The COCO-based datasets reference COCO 2014 train images by id and file name; download
them from [cocodataset.org](https://cocodataset.org).
