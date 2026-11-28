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

| Live model comparison | Latency and throughput |
| --- | --- |
| <a href="material/chay-tung-fig-evit-vit.mp4"><img src="material/ViT-EViT-compare.png" width="440" alt="One X-ray shown alongside ground truth and both model predictions"></a> | <a href="material/vid-compare-vit-evit.mp4"><img src="material/latency.png" width="440" alt="Comparison of inference latency, p95 latency, and throughput"></a> |
| [Watch both models segment successive images · 0:16](material/chay-tung-fig-evit-vit.mp4) | [Watch inference timing and throughput update · 0:52](material/vid-compare-vit-evit.mp4) |

<details>
<summary>More evaluation charts</summary>

| Aggregate accuracy | Dice per tumor image |
| --- | --- |
| <a href="material/accuracy.png"><img src="material/accuracy.png" width="440" alt="Pooled Dice, pooled IoU, and false positives on normal images"></a> | <a href="material/paired-dice.png"><img src="material/paired-dice.png" width="224" alt="Scatter plot of ViT and EfficientViT Dice on each tumor image"></a> |
| **Parameters and GFLOPs** | **Runtime resources** |
| <a href="material/complexity.png"><img src="material/complexity.png" width="320" alt="Model parameter count and GFLOPs per forward pass"></a> | <a href="material/resources.png"><img src="material/resources.png" width="440" alt="RAM and CPU measurements, with VRAM unavailable in this CPU run"></a> |

</details>

## Architecture

The local Python HTTP server reads the BTXRD archive, rasterizes polygon annotations, and runs PyTorch inference. The HTML, CSS, and JavaScript frontend fetches images, masks, and metrics through the API. SQLite stores run history in `outputs/demo_history.sqlite3`.

<details>
<summary>Repository tree</summary>

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

</details>

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
