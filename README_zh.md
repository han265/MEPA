# MEPA：面向视觉自回归建模的多尺度表征对齐与混合专家方法

<p align="center">
  <a href="README.md">English</a> | <strong>中文</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.00371">
    <img src="https://img.shields.io/badge/arXiv-2607.00371-b31b1b.svg" alt="arXiv:2607.00371">
  </a>
  <a href="https://arxiv.org/pdf/2607.00371">
    <img src="https://img.shields.io/badge/Paper-PDF-blue.svg" alt="Paper PDF">
  </a>
  <img src="https://img.shields.io/badge/ECCV-2026-4c1.svg" alt="ECCV 2026">
  <img src="https://img.shields.io/badge/Task-Image%20Generation-ff69b4.svg" alt="Image Generation">
</p>

<p align="center">
  Nuoyan Zhou, Zhijun Tu, Lei Yu, Kun Cheng, Jie Hu, Nannan Wang, Xinghao Chen
  <img src="assets/framework.png" width="96%">
</p>

本仓库是 **Multi-Scale Representation Alignment for Visual Autoregressive Modeling with Mixture of Experts（MEPA）** 的复现实现。MEPA 是一项 **ECCV 2026** 工作，旨在提升 Visual AutoRegressive（VAR）模型的**训练效率**与**生成质量**。

## 主要贡献

VAR 通过 next-scale prediction 开创了由粗到细的视觉自回归生成范式。论文指出，原始 VAR 仍存在两个关键问题：

- **尺度冲突**：低尺度主要学习全局语义，高尺度更关注细粒度细节；所有尺度共享同一套稠密 FFN 结构会带来优化目标冲突。
- **语义错误传播**：由于后续尺度依赖前面尺度的预测，小尺度或中间尺度中的语义错误会被逐级放大，最终影响生成图像质量。

MEPA 通过以下两个核心设计缓解上述问题：

1. **Scale-aware Token-routed Mixture-of-Experts（STMoE）**：基于尺度感知 token 路由的 MoE 模块，用于解耦多尺度表征学习。
2. **Semantic Guidance（SG）**：利用预训练自监督视觉编码器提供语义监督，强化小尺度与中间尺度的语义建模能力。

<p align="center">
  <img src="assets/motivation_new1.png" width="88%">
</p>

## 安装

1. 安装 PyTorch。原始 VAR 代码要求 `torch>=2.0`；本复现代码的 `requirements.txt` 中使用的是 `torch~=2.1.0`。
2. 安装 Python 依赖：

```bash
pip install -r requirements.txt
```

可选安装加速包：

```bash
pip install flash-attn xformers
```

如果安装了这些包，代码会在可用时自动使用更快的 attention / MLP fused kernel。

## 数据集

请将 ImageNet-1K 准备为标准文件夹格式：

```text
/path/to/imagenet/
  train/
    n01440764/
      *.JPEG
    ...
  val/
    n01440764/
      *.JPEG
    ...
```

训练时通过 `--data_path=/path/to/imagenet` 指定数据集路径。


## 预训练权重

  ~./code/
  ├── MEPA-main/
  │   ├── train.py
  │   ├── models/
  │   ├── vae_ch160v4096z32.pth (download from https://huggingface.co/FoundationVision/var/)
  │   │   ├── dinov3_loader.py
  │   │   └── ...
  │   └── ...
  ├── dinov3/ (git clone https://github.com/facebookresearch/dinov3.git)
  │   ├── hubconf.py
  │   ├── dinov3/
  │   └── ...
  └── pretrained_models/
      ├── your_dinov3_weight_file.pth (hf download facebook/dinov3-vitb16-pretrain-lvd1689m)

## 训练

```bash
# MEPA-d16, ImageNet 256×256, 100 epochs
torchrun --nproc_per_node=8 train.py \
  --data_path=/path/to/imagenet \
  --depth=16 \
  --pn=256 \
  --bs=96 \
  --ep=100 \
  --tblr=1e-4 \
  --twd=0.05 \
  --wpe=0.1 \
  --fp16=1 \
  --alng=1e-3 \
  --enc_type=dinov3-vit-b16 \
  --align_start=5 \
  --align_end=7

# MEPA-d12：使用 --depth=12

# MEPA-d20, ImageNet 512×512, 200 epochs
torchrun --nproc_per_node=8 train.py \
  --depth=20 \
  --saln=1 \
  --pn=512 \
  --bs=96 \
  --ep=200 \
  --tblr=8e-5 \
  --fp16=1 \
  --alng=5e-6 \
  --wpe=0.01 \
  --twde=0.08 \
  --enc_type=dinov3-vit-b16 \
  --align_start=5 \
  --align_end=7
```

## 采样与评估

采样接口沿用原始 VAR API。对于 ImageNet 类别条件生成评估，可使用：

```python
var.autoregressive_infer_cfg(
    B=B,
    label_B=label_B,
    cfg=1.5,
    top_p=0.96,
    top_k=900,
    more_smooth=False,
)
```

评估 FID / IS / Precision / Recall 时，请生成 50,000 张 PNG 图像，并使用原始 VAR 评估协议中的 ImageNet reference statistics 进行评估。

## 实现说明

本代码库主要修改了以下模块：

| 文件 | 主要改动 |
|---|---|
| `models/moe.py` | 实现 `MoEGate`、负载均衡辅助损失，以及包含 routed FFN experts 和 shared expert 的 `SparseMoeBlock`。 |
| `models/basic_var.py` | 新增 `AdaLNSelfAttn_MoE`，用 `SparseMoeBlock` 替换 VAR block 中原始的稠密 FFN 分支。 |
| `models/var.py` | 默认使用 MoE block；新增中间层 projector，将 VAR hidden states 映射到 768 维特征空间用于语义对齐。 |
| `utils/arg_util.py` | 新增 MEPA 相关参数：`--enc_type`、`--align_start`、`--align_end`。 |
| `utils/load_encoder.py` | 提供统一的预训练视觉编码器封装，支持 DINO / DINOv2 / DINOv3 / MoCoV3 / MAE / CLIP / SigLIP 等。 |
| `train.py` | 加载外部视觉编码器，并传入 `VARTrainer`。 |
| `trainer.py` | 在原始 VAR token cross-entropy loss 之外，计算选定残差尺度上的 semantic alignment loss。 |

补充说明：

- 本仓库是复现分支，并非原始 VAR 官方发布代码。
- `utils/arg_util.py` 中默认 `--enc_type` 当前为 `dinov3`；而 `utils/load_encoder.py` 期望传入完整的三段式格式，例如 `dinov3-vit-b` 或 `dinov2-vit-b`。
- 如果更换外部编码器的特征维度，需要同步修改 `models/var.py` 中 `projector` 的输出维度，以及 `trainer.py` 中的对齐 reshape 逻辑。
- STMoE 的超参数目前在 `models/moe.py` 中硬编码：`num_experts=8`、`num_experts_per_tok=2`、`aux_loss_alpha=0.01`。
- 当前 MoE 实现同时包含 shared expert 与 routed experts，对应论文中保留通用能力并引入专家特化能力的设计思想。

## 论文主要结果

<p align="center">
  <img src="assets/pictures.png" width="80%">
</p>

### ImageNet 256×256

| Model | FID ↓ | IS ↑ | Precision ↑ | Recall ↑ | Params | Steps | Epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAR-d16 | 3.55 | 280.4 | 0.84 | 0.51 | 310M | 10 | 200 |
| VAR-d20 | 2.95 | 302.6 | 0.83 | 0.56 | 600M | 10 | 250 |
| FlexVAR-d16 | 3.05 | 291.3 | 0.83 | 0.52 | 310M | 10 | 180 |
| FlexVAR-d20 | 2.41 | 299.3 | 0.85 | 0.58 | 600M | 10 | 250 |
| SpectralAR-d16 | 3.02 | 282.2 | 0.81 | 0.55 | 310M | 64 | 200 |
| SpectralAR-d20 | 2.49 | 305.4 | - | - | 600M | 64 | 250 |
| **MEPA-d12** | **2.86** | **288.44** | 0.82 | 0.55 | 252M | 10 | 200 |
| **MEPA-d16** | **2.32** | **311.26** | 0.82 | 0.57 | 585M | 10 | 200 |
| VAR-d16 reproduced | 4.10 | 241.6 | 0.85 | 0.47 | 310M | 10 | 100 |
| FlexVAR-d16 reproduced | 7.68 | 176.63 | 0.77 | 0.49 | 310M | 10 | 100 |
| FlexVAR-d20 reproduced | 5.77 | 215.04 | 0.77 | 0.52 | 600M | 10 | 100 |
| **MEPA-d12** | **3.27** | **260.97** | 0.81 | 0.53 | 252M | 10 | 100 |
| **MEPA-d16** | **2.65** | **304.60** | 0.82 | 0.56 | 585M | 10 | 100 |

### ImageNet 512×512

由于计算资源有限，论文中的 MEPA-d20 仅训练了 66 个 epoch，尚未完全收敛。

| Model | BigGAN | ADM | DiT-XL/2 | MaskGIT | VQGAN | VAR-d36-s | **MEPA-d20** |
|---|---:|---:|---:|---:|---:|---:|---:|
| FID ↓ | 8.43 | 23.24 | 3.04 | 7.32 | 26.52 | **2.63** | 7.29 |
| IS ↑ | 177.9 | 101.0 | 240.8 | 156.0 | 66.8 | **303.2** | 199.2 |
| Epoch | - | - | - | - | - | 350 | 66 |

## 致谢

本仓库基于优秀的 [VAR](https://github.com/FoundationVision/VAR)、[REPA](https://github.com/sihyun-yu/REPA)、[DiT-MoE](https://github.com/feizc/DiT-MoE) 代码库构建。感谢 Codex 撰写了这个高质量的 README 文件。

## 引用

如果本复现代码或 MEPA 论文对你的研究有帮助，请引用：

```bibtex
@misc{mepa2026,
  title         = {Multi-Scale Representation Alignment for Visual Autoregressive Modeling with Mixture of Experts},
  author        = {Zhou, Nuoyan and Tu, Zhijun and Yu, Lei and Cheng, Kun and Hu, Jie and Wang, Nannan and Chen, Xinghao},
  year          = {2026},
  eprint        = {2607.00371},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2607.00371}
}
```
