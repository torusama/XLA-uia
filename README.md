# Vision Transformer and EfficientViT for Medical Image Analysis

This repository supports a seminar project surveying Vision Transformers (ViT) for image classification and EfficientViT for image segmentation. The planned experiments focus on medical X-ray images, starting with fracture classification on FracAtlas.

The repository currently provides a research scaffold and a PyTorch ViT-B/16 model adapter. Dataset preparation, training, evaluation, and segmentation are planned and have not yet been implemented.

## Research goals

- Study how ViT uses image patches and self-attention for classification.
- Study EfficientViT by Han Cai et al. (ICCV 2023) and its approach to efficient dense prediction.
- Compare classification models on the same dataset and splits, reporting predictive performance and computational cost under a consistent protocol.
- Explore segmentation separately, using the same segmentation data and ground truth for each model being compared. Classification accuracy and segmentation Dice/IoU measure different tasks and should not be compared directly.

## What is implemented

- A Python package with configuration files and a CLI for checking the environment and experiment configuration.
- A ViT-B/16 adapter using the `timm` model `vit_base_patch16_224.orig_in21k`, with a two-class classification head.
- CPU tests for model structure, forward output and head gradients, plus configuration and CLI checks.
- Documentation and placeholders for data processing, training, metrics, benchmarking, and future models.

The ViT backbone uses the PyTorch port provided by `timm`; the original Google implementation is JAX/Flax. The pretrained backbone was trained on ImageNet-21k. Its two-class head is newly initialized and **has not been trained on fracture images**, so its predictions are not meaningful fracture classifications. See the [model card](https://huggingface.co/timm/vit_base_patch16_224.orig_in21k) and [source review](docs/vit/source_review.md).

## Setup

Python 3.10 or 3.11 is recommended. Install the appropriate CPU or CUDA build of PyTorch for your machine using the [official instructions](https://pytorch.org/get-started/locally/), then install the project and its development and ViT dependencies.

Windows PowerShell:

```powershell
$env:PYTHONUTF8 = "1"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# Install the appropriate PyTorch and torchvision build first.
python -m pip install -e ".[dev,vit]"
```

Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
# Install the appropriate PyTorch and torchvision build first.
python -m pip install -e '.[dev,vit]'
```

On Windows, set `PYTHONUTF8=1` in each PowerShell session when working with non-ASCII paths. Installation requires access to a Python package index. A GPU is optional for the current checks.

## Check the installation

Run these commands from the repository root:

```text
python -m xla --help
python -m xla doctor
python -m xla config-check --config configs/experiments/fracatlas_classification.yaml
python -m pytest -q
python -m ruff check .
```

The checks do not require FracAtlas or pretrained weights. Tests using randomly initialized models are technical checks, not experimental results. The config checker reports fields marked `TODO_VERIFY` that must be confirmed before a real experiment.

## Use the ViT adapter

```python
from xla.models.vit import build_vit, preprocessing_config

model = build_vit(pretrained=True, num_classes=2)
print(preprocessing_config(model))
```

The first pretrained load downloads weights if they are not already cached. The adapter returns a PyTorch model that accepts tensors of shape `[batch, 3, 224, 224]` and produces logits of shape `[batch, 2]`. The returned preprocessing configuration describes the pretrained model's defaults; an X-ray preprocessing policy still needs to be validated before training. Use `pretrained=False` only when random initialization is intended, such as offline tests.

## Project layout

```text
configs/       Dataset, model, and experiment configuration
src/xla/       CLI, ViT adapter, and modules reserved for future work
tests/         Offline unit and smoke tests
docs/          Research scope, protocols, references, and ViT notes
data/          Local dataset and split locations; datasets are not included
outputs/       Local experiment artifacts; results are not included
```

## Next steps

1. Obtain and inspect FracAtlas, verify labels and image integrity, and create leakage-aware fixed splits.
2. Implement training and evaluation for the ViT classifier using those splits.
3. Integrate EfficientViT and evaluate models under documented, comparable conditions; develop segmentation as a separate task.

See the [project scope](docs/project_scope.md), [data protocol](docs/data_protocol.md), [experiment protocol](docs/experiment_protocol.md), [ViT walkthrough](docs/vit/code_walkthrough.md), and [source registry](docs/references/source_registry.md) for further details.
