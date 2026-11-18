# Glance

**一次前向，对图像或文本给出校准过的类型化决策。**

[English](README.md) · [用法](#用法) · [基准](README.md#benchmarks) · [Hugging Face 权重](https://huggingface.co/aac6fef/Glance-4B-preview) · [更新日志](CHANGELOG.md)

Glance 回答类型化问题：找到某个物体、从选项中挑一个、判断是或否、按等级打分，每个答案都带校准过的概率。图像或文本只读一次，所有问题都由这一次 prefill 回答，不逐 token 生成。目标不在图里时，它会回答"不存在"并给出概率。在 RTX PRO 6000 上，对一张 COCO 图片问 8 个问题约 **86 ms**，和问 1 个一样；生成式的基座模型需要 3.2 s。

![Glance 对每张图或每段文本一次前向回答 4 到 6 个问题（真实模型输出）](https://raw.githubusercontent.com/Vision-Decision/Glance/main/assets/demo.gif)

## 快速开始

```bash
pip install "glance @ git+https://github.com/Vision-Decision/Glance"
pip install flash-linear-attention   # 可选，仅 CUDA：Gated-DeltaNet 快速内核
```

```python
from glance import Glance

m = Glance.from_pretrained("aac6fef/Glance-4B-preview")   # 依次尝试 cuda、mps、cpu
r = m.box("assets/examples/park_bench.jpg", "the clear umbrella")
print(r.present_prob, r.box, r.confidence)                 # 0.996 (189.2, 9.3, 373.6, 164.8) 0.8233
```

首次调用会下载 Qwen3.5-4B 基座权重（约 9 GB）以及 Glance 的 adapter 和 heads（约 537 MB）。

## 四种原语

| 原语 | 输入 | 输出 |
|---|---|---|
| `box` | 指代表达 | P(存在)、像素框、P(IoU ≥ 0.5)，以及框置信度 = P(存在) · P(IoU ≥ 0.5) |
| `choice` | 问题和 2–26 个选项 | 每个选项的概率 |
| `yesno` | 是非问题 | P(是) |
| `score` | 问题和有序等级 | 每个等级的概率和期望等级 |

## 用法

```python
from glance import Glance, Choice, YesNo, Score

m = Glance.from_pretrained("aac6fef/Glance-4B-preview")
m.box("assets/examples/park_bench.jpg", "a red bicycle")          # 图中没有
# BoxResult('a red bicycle', present_prob=0.000, box=[5, 171, 236, 363], confidence=0.000)
m.yesno("assets/examples/editor.png", "Is a save dialog open?")
# YesNoResult(p_yes=0.992)

# 一次共享前缀前向回答多个问题；纯字符串表示 box 问题
rs = m.ask("assets/examples/vegetables.jpg", [
    "the broccoli", "a banana",
    YesNo("Is this food cooked?"),
    Choice("What is the main food group?", ["vegetables", "meat", "dairy", "grains"]),
    Score("How healthy is this food?", ["unhealthy", "neutral", "healthy", "very healthy"]),
])
```

文本也一样，把字符串作为状态传入即可，例如 `m.choice(ticket, "Which team should handle this ticket?", [...])`。截图建议设 `max_vis_tokens=2048`。更多示例、真实输出和完整说明见 [英文 README](README.md#usage)。

## 基准

![基准对比](assets/benchmarks.png)

Glance-4B-preview 在"目标不存在"的判断和拒答、校准以及速度上领先，指代定位也超过生成式基座。它在 ScreenSpot-Pro 和 VenusBench-GD 定位上明显落后于 GUI 专用模型。数据来源、评测协议和脚注见 [英文 README](README.md#benchmarks)。

## 许可

- 本仓库代码：Apache-2.0。
- 权重（`aac6fef/Glance-4B-preview`）：**CC BY-NC 4.0，仅限非商业用途**。训练数据包含仅限非商用的 gRefCOCO，以及部分许可无法确认的文本数据源。
- 基座模型 Qwen/Qwen3.5-4B：Apache-2.0。
