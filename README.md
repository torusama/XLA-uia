<h1 align="center">BoneScope</h1>

<h3 align="center">Bone Tumor Segmentation &amp; Model Comparison Workspace</h3>

<p align="center">
  Explore bone tumor masks on X-rays and compare ViT and EfficientViT predictions, accuracy, and inference performance on BTXRD.
</p>

<p align="center">
  <a href="README.md"><img alt="Language: English" src="https://img.shields.io/badge/LANGUAGE-ENGLISH-2563eb?style=for-the-badge"></a>
  <a href="README.vi.md"><img alt="Ngôn ngữ: Tiếng Việt" src="https://img.shields.io/badge/NG%C3%94N%20NG%E1%BB%AE-TI%E1%BA%BENG%20VI%E1%BB%86T-20604a?style=for-the-badge"></a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?logo=python&amp;logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&amp;logoColor=white">
  <img alt="JavaScript" src="https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&amp;logoColor=black">
  <img alt="SQLite" src="https://img.shields.io/badge/SQLite-003B57?logo=sqlite&amp;logoColor=white">
  <img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-147CB3?logo=kaggle&amp;logoColor=white">
</p>

<p align="center"><strong>Digital Image Processing Seminar · BTXRD Bone Tumor Segmentation</strong></p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#features">Features</a> ·
  <a href="#gallery">Gallery</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#documentation">Documentation</a>
</p>

---

## Overview

BoneScope compares **Vision Transformer (ViT-B/16)** and **EfficientViT-Seg-B1** for binary bone tumor segmentation. Both models predict a foreground mask from a 224 × 224 X-ray. The repository contains the training code, Kaggle notebooks, and a local browser demo that shows model predictions alongside ground truth.

## Features

- Browse 561 test images with polygon-derived ground truth and model overlays.
- Compare Dice, IoU, latency, throughput, RAM, CPU, VRAM, parameter count, and GFLOPs on the same image.
- Run both models on the same 560 quality-checked test images, track progress, and stop after the current image.
- After a completed run, view the aggregate charts and download a ZIP with PNG charts, per-image CSV, summary data, and the exact test-image selection.
- Filter SQLite-backed inference history by session or batch run, expand charts, and export CSV.

## Gallery

| ViT segmentation | EfficientViT segmentation |
| --- | --- |
| <a href="material/ViT-seg.png"><img src="material/ViT-seg.png" width="440" alt="ViT demo showing a knee X-ray, ground-truth mask, and predicted tumor overlay"></a> | <a href="material/EViT-seg.png"><img src="material/EViT-seg.png" width="440" alt="EfficientViT demo showing a shoulder X-ray, ground-truth mask, and predicted tumor overlay"></a> |

### Live segmentation · 0:16

https://github.com/user-attachments/assets/24bda40b-7fbb-4bfb-87da-bf70212f1c23

### Inference history · 0:52

https://github.com/user-attachments/assets/a72c0550-4f4a-45c3-bf21-6f7f0406bba8

### Evaluation charts

| Aggregate accuracy | Dice per tumor image |
| --- | --- |
| <a href="material/accuracy.png"><img src="material/accuracy.png" width="440" alt="Pooled Dice, pooled IoU, and false positives on normal images"></a> | <a href="material/paired-dice.png"><img src="material/paired-dice.png" width="224" alt="Scatter plot of ViT and EfficientViT Dice on each tumor image"></a> |
| **Latency and throughput** | **Runtime resources** |
| <a href="material/latency.png"><img src="material/latency.png" width="440" alt="Latency distribution, p95 latency, and forward throughput"></a> | <a href="material/resources.png"><img src="material/resources.png" width="440" alt="RAM and CPU measurements, with VRAM unavailable in this CPU run"></a> |
| **Parameters and GFLOPs** | **Side-by-side predictions** |
| <a href="material/complexity.png"><img src="material/complexity.png" width="320" alt="Model parameter count and GFLOPs per forward pass"></a> | <a href="material/ViT-EViT-compare.png"><img src="material/ViT-EViT-compare.png" width="440" alt="One X-ray with ground truth and predictions from both models"></a> |

## Architecture

Both models take a 224 × 224 RGB X-ray with ImageNet normalization. The demo resizes each model's logits to 224 × 224, applies sigmoid, and thresholds at 0.5. The backend compares the masks against the rasterized BTXRD polygons, sends overlays and metrics to the browser, and stores measurements in SQLite.

### ViT-B/16 with convolutional skip connections

```mermaid
flowchart LR
    I["RGB X-ray · 224×224×3<br/>ImageNet normalization"]
    subgraph V["ViT-B/16 encoder"]
        direction TB
        P["16×16 patch projection<br/>196 tokens × 768"] --> T["CLS + position embeddings<br/>12 Transformer blocks · 12 heads"]
        T --> C["Drop CLS · reshape<br/>1×1 projection · 14×14×128"]
    end
    subgraph S["Convolutional skips"]
        direction TB
        S1["Stride-2 conv · 112×112×24"] --> S2["Stride-2 conv · 56×56×48"]
        S2 --> S3["Stride-2 conv · 28×28×64"]
    end
    subgraph D["Decoder"]
        direction TB
        D1["Resize + concat 28×28 skip<br/>3×3 conv · 28×28×96"] --> D2["Resize + concat 56×56 skip<br/>3×3 conv · 56×56×64"]
        D2 --> D3["Resize + concat 112×112 skip<br/>3×3 conv · 112×112×32"]
        D3 --> L["1×1 mask head · 112×112×1 logits"]
    end
    I --> V --> D
    I --> S --> D
    D --> O["Bilinear resize · sigmoid<br/>224×224 mask · threshold 0.5"]
```

The encoder models relationships between patches. The decoder fuses that context with higher-resolution features from the image at three scales. Each detail and fusion block uses a 3×3 convolution, GroupNorm, and GELU. [ViT implementation](src/xla/models/vit_detail.py)

### EfficientViT-Seg-B1

```mermaid
flowchart LR
    I["RGB X-ray · 224×224×3<br/>ImageNet normalization"]
    subgraph B["EfficientViT-B1 backbone"]
        direction TB
        S01["Stem + stage 1<br/>112×112×16 → 56×56×32"] --> S2["Stage 2 · 3 MBConv<br/>28×28×64"]
        S2 --> S3["Stage 3 · 3 EfficientViT blocks<br/>14×14×128"]
        S3 --> S4["Stage 4 · 4 EfficientViT blocks<br/>7×7×256"]
    end
    subgraph H["Segmentation head"]
        direction TB
        P["1×1 projection at stages 2, 3, 4<br/>Resize by 1×, 2×, 4×"] --> F["Elementwise sum · 28×28×64"]
        F --> M["3 residual MBConv"]
        M --> L["1×1 mask head · 28×28×1 logits"]
    end
    I --> B --> H --> O["Bilinear resize · sigmoid<br/>224×224 mask · threshold 0.5"]
```

Stages 3 and 4 each start with a downsampling MBConv. The EfficientViT blocks combine lightweight multiscale linear attention (LiteMLA) with MBConv. The segmentation head aligns stages 2–4 at stride 8 and adds their features before predicting one foreground logit per location. [Demo adapter](demo/backend/models.py) · [Pinned upstream segmentation head](https://github.com/mit-han-lab/efficientvit/blob/de7d7733cc0329f391b33f1f459271562ec27bd5/efficientvit/models/efficientvit/seg.py)

### Repository tree

```text
start-web.ps1      Windows web launcher
requirements.txt   Python dependencies for the demo and training
demo/              Browser UI, HTTP API, and SQLite storage
src/xla/           Data loaders, models, training loops, and Segmenter code
scripts/           Data preparation, training, and notebook packaging
notebooks/         Kaggle jobs and metadata
configs/           Model, dataset, and experiment configuration
data/              Local datasets and split manifests
outputs/           Checkpoints, Kaggle results, and demo history
docs/              Data protocol, experiments, and references
material/          Demo screenshots and video clips
paper/             Research papers
tests/             Python tests
```

## Quick Start

Install Python 3.10 or newer. From the repository root on Windows, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start-web.ps1
```

Open **http://127.0.0.1:8765** when the server reports that it is ready. Press **Ctrl+C** to stop it. Add `-Port 8766` if the default port is busy.

The launcher creates or reuses a Python environment and keeps compatible packages, including existing CPU or CUDA PyTorch builds. On the first run it downloads the BTXRD archive, test manifest, and both model checkpoints from Kaggle (about 410 MB total, plus any missing Python packages). It checks SHA-256 and reuses verified files on later runs. **Internet access is required for the first run. A Kaggle account or CLI is optional.** [Python requirements](requirements.txt)

| Public Kaggle release | Contents |
| --- | --- |
| [BTXRD dataset and splits](https://www.kaggle.com/datasets/vtan2405/bonescope-btxrd-splits) | Image and mask annotation archive, train/validation/test manifests, and the matched 1,302-image tumor-only training manifest |
| [ViT and EfficientViT checkpoints](https://www.kaggle.com/datasets/vtan2405/bonescope-btxrd-checkpoints) | The two PyTorch weight files used by the web demo |

| Demo model | Default checkpoint |
| --- | --- |
| ViT-B/16 with convolutional skip connections | `outputs/kaggle_checks/vit-detail/best_vit_detail.pt` · epoch 32 · threshold 0.5 |
| EfficientViT-Seg-B1 | `outputs/kaggle_checks/efficientvit/best_evit_seg_btxrd.pth` · threshold 0.5 |

The archive is cached under `data/raw/` and the test manifest under `data/splits/`. Git ignores downloaded weights, dataset files, and generated reports. To override a checkpoint, set `XLA_VIT_CHECKPOINT` or `XLA_EFFICIENTVIT_CHECKPOINT`.

## Evaluation

### Dataset

Kaggle version 2 stores the exact `btxrd_final.zip` archive as `btxrd_final.bin` to preserve its checksum. The launcher restores the `.zip` extension locally. The archive comes from Hugging Face revision `cf8b0569b18e43e7cecab412a06e60aeaf258b32`. The manifests exclude byte-identical images with conflicting labels and training images duplicated in held-out sets. The original BTXRD dataset is published by Yao et al. under CC BY 4.0. [Original dataset record](https://api.figshare.com/v2/articles/27865398) · [Split protocol](docs/btxrd_data.md)

| Split after filtering | Normal | Tumor | Total |
| --- | ---: | ---: | ---: |
| Train | 1,311 | 1,302 | 2,613 |
| Validation | 282 | 280 | 562 |
| Test | 281 | 280 | 561 |

### Kaggle results

The head-to-head experiment used the same **1,302 tumor-positive training images**, full validation and test sets, and threshold 0.5 with horizontal-flip test-time augmentation. EfficientViT led by 3.53 percentage points in Dice and 4.33 points in IoU. The earlier Mask Transformer score comes from validation. [Comparison output](https://www.kaggle.com/code/vtan2405/xla-btxrd-vit-detail-vs-efficientvit/output)

| Experiment | Split | Pixel micro Dice | Pixel micro IoU |
| --- | --- | ---: | ---: |
| EfficientViT-Seg-B1 | Test | **0.7404** | **0.5878** |
| ViT-B/16 with convolutional skip connections | Test | 0.7051 | 0.5445 |
| ViT-B/16 + Mask Transformer, threshold 0.7 | Validation | 0.6238 | 0.4533 |

The demo uses one forward pass per model per image. It reports per-image Dice and IoU. When both masks are empty, each score is 1. Batch runs also report pixel-pooled Dice and IoU. Latency measures the forward pass. RAM uses process RSS, VRAM uses PyTorch tensor allocations, and GFLOPs reflect operators supported by the PyTorch counter.

Before a demo batch run, the backend checks image decoding, dimensions, SHA-256, annotation and mask consistency, and byte-level duplicates. The current selection contains 279 tumor images and 281 normal images. The browser and ZIP share the same Matplotlib charts.

## Documentation

| Guide | Contents |
| --- | --- |
| [BTXRD data protocol](docs/btxrd_data.md) | Dataset revision, duplicate filtering, and split generation |
| [Experiments](docs/experiments.md) | Training configurations, Kaggle notebooks, and results |
| [Sources and licenses](docs/sources.md) | Papers, models, and upstream licenses |

### Development

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m xla doctor
.\.venv\Scripts\python.exe -m pytest -q
```

Run `python -m demo.build_site` with the local dataset to build a static demo with images and ground truth.
