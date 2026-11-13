# Kh?o s?t ViT v? EfficientViT

??y l? n?n Phase 0 cho seminar: package, config, CLI v? t?i li?u; **ch?a c? data loader, m? h?nh, training hay k?t qu? th?c nghi?m**. Nh?nh ch?nh d? ki?n ph?n lo?i FracAtlas; segmentation l? h??ng m? r?ng, ??nh gi? ri?ng.

## C?i ??t

Windows PowerShell (Python 3.10/3.11):

```powershell
$env:PYTHONUTF8 = "1"  # Set in each PowerShell session
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Linux/Colab:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Tr?n Windows, gi? `PYTHONUTF8=1` trong phi?n PowerShell ?? tr?nh l?i m? h?a ???ng d?n v? output c? ti?ng Vi?t.

C?i ??t c?n truy c?p package index; sau khi c?i, c?c ki?m tra sau ch?y offline, kh?ng c?n GPU hay dataset:

```text
python -m xla --help
python -m xla doctor
python -m xla config-check --config configs/experiments/fracatlas_classification.yaml
python -m pytest -q
python -m ruff check .
```

Torch l? t?y ch?n: xem [h??ng d?n CPU/CUDA](requirements/README-cuda.md). Xem [??c t?](docs/CODEX_SETUP_SPEC.md), [ki?n tr?c](docs/architecture.md), [ngu?n](docs/references/source_registry.md) v? [roadmap](docs/roadmap.md).

## ViT-B/16 PyTorch

C?i PyTorch ph? h?p ph?n c?ng theo [h??ng d?n ch?nh th?c](https://pytorch.org/get-started/locally/), sau ?? ch?y `python -m pip install -e ".[dev,vit]"`. V?i m?y hi?n t?i ?? th? b?n CPU; GPU NVIDIA driver c?n ki?m tra t??ng th?ch tr??c khi chuy?n sang CUDA.

```python
from xla.models.vit import build_vit, preprocessing_config

model = build_vit(pretrained=True)  # T?i checkpoint n?u ch?a c? cache
print(preprocessing_config(model))
```

Backbone d?ng b?n port c?a `timm`; head 2 l?p ???c kh?i t?o m?i. Ch?a c? dataset v? ch?a tri?n khai training, evaluation ho?c prediction. `pretrained=False` ch? d?ng cho ki?m tra random initialization. Xem [gi?i th?ch thu?t to?n](docs/vit/code_walkthrough.md) v? [ngu?n](docs/vit/source_review.md).
