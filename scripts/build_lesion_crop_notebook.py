"""Package the tested crop A/B sources into a self-contained private Kaggle notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = [
    "src/xla/__init__.py", "src/xla/data/__init__.py",
    "src/xla/data/hf_segmentation.py", "src/xla/data/augmentation.py",
    "src/xla/data/lesion_crop.py", "src/xla/engine/__init__.py",
    "src/xla/engine/vit_segmentation.py", "src/xla/engine/segmentation_audit.py",
    "src/xla/models/__init__.py", "src/xla/models/vit.py", "src/xla/models/segmenter_vit.py",
    "src/xla/vendor/__init__.py", "src/xla/vendor/segmenter/__init__.py",
    "src/xla/vendor/segmenter/blocks.py", "src/xla/vendor/segmenter/decoder.py",
    "src/xla/vendor/segmenter/utils.py", "src/xla/vendor/segmenter/LICENSE",
    "scripts/improve_segmenter.py",
]


def build_notebook():
    sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in SOURCE_PATHS}
    cells = []

    def cell(kind, source):
        entry = {"cell_type": kind, "metadata": {}, "id": f"crop-ab-{len(cells)}",
                 "source": source.splitlines(keepends=True)}
        if kind == "code":
            entry.update(execution_count=None, outputs=[])
            compile(source, entry["id"], "exec")
        cells.append(entry)

    cell("markdown", """# BTXRD ViT: small-lesion crop A/B

Compare full-image paired augmentation against the same augmentation followed by
training-only small-lesion crops. Both arms start from the pinned weight-10
checkpoint and use ViT-B/16 + Mask Transformer, batch Dice + BCE foreground weight
10, seed 42, batch 4, AdamW learning rate 2e-6, weight decay 0.01 and 20 epochs.

For an augmented positive mask with 1-255 foreground pixels, crop with probability
0.5; otherwise keep the full image. A square crop retains the entire foreground
and up to 8 pixels of context on each side (clipped to the image boundary).
Crop sides are 112-196 pixels unless a larger side is needed to retain the mask;
if no crop can fit, keep the full image. Resize the pair back to 224x224 using
bilinear image interpolation and nearest mask interpolation. Normal and larger
positive cases remain full images; no cases are dropped or duplicated.

Validation always uses uncropped full images. Test is not evaluated. This is a
controlled experiment, not a claim that crop training improves segmentation.
""")
    cell("code", """import os
import subprocess
import sys
from pathlib import Path

assert Path('/kaggle/working').is_dir(), 'This notebook must run on Kaggle.'
os.environ['HF_HOME'] = '/kaggle/temp/huggingface'
os.environ['PYTHONUNBUFFERED'] = '1'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'timm==1.0.31', 'einops==0.8.1'], check=True)
import timm
import torch
from huggingface_hub import hf_hub_download

assert torch.cuda.is_available(), 'CUDA unavailable: check Kaggle accelerator and GPU quota.'
print('Torch:', torch.__version__, 'timm:', timm.__version__)
print('GPU:', torch.cuda.get_device_name(0), flush=True)
""")
    cell("code", """import hashlib

DATASET_REVISION = 'cf8b0569b18e43e7cecab412a06e60aeaf258b32'
EXPECTED_SHA256 = '59ce94244fdf997f0c68e6a378690a4359e6c411f7fda7ac2f3994489637884b'
archive = Path(hf_hub_download(repo_id='torusama/xla-seminar-uia', repo_type='dataset',
                               filename='btxrd_final.zip', revision=DATASET_REVISION))
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
if digest != EXPECTED_SHA256:
    raise RuntimeError(f'Dataset checksum mismatch: {digest}')
print('BTXRD downloaded; SHA-256:', digest, flush=True)
""")
    cell("code", "# Frozen tested sources and the upstream MIT license.\n"
         "PROJECT = Path('/kaggle/temp/xla-vit-lesion-crop-ab')\n"
         f"SOURCES = {sources!r}\n"
         "for name, content in SOURCES.items():\n"
         "    path = PROJECT / name\n"
         "    path.parent.mkdir(parents=True, exist_ok=True)\n"
         "    path.write_text(content, encoding='utf-8')\n"
         "os.environ['PYTHONPATH'] = str(PROJECT / 'src')\n"
         "print('Prepared project sources:', len(SOURCES), flush=True)\n")
    cell("code", """SOURCE_SHA256 = 'ea9eb43856adb5fd20a4d430aa2afb2dedf6156fa9f76e1d458ff6297ec1a056'
checkpoint = None
for candidate in sorted(Path('/kaggle/input').rglob('best.pt')):
    if candidate.parent.name != 'weight_10':
        continue
    with candidate.open('rb') as stream:
        candidate_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    if candidate_hash == SOURCE_SHA256:
        checkpoint = candidate
        break
if checkpoint is None:
    raise FileNotFoundError('Pinned weight_10/best.pt missing from the attached notebook output')
output = Path('/kaggle/working/vit_lesion_crop_ab')
command = [sys.executable, '-u', str(PROJECT / 'scripts/improve_segmenter.py'),
           '--archive', str(archive), '--checkpoint', str(checkpoint), '--output', str(output),
           '--expected-source-sha256', SOURCE_SHA256,
           '--epochs', '20', '--weights', '10', '--crop-trial', '--crop-probability', '0.5',
           '--batch-size', '4', '--learning-rate', '0.000002',
           '--workers', '2', '--seed', '42', '--device', 'cuda']
subprocess.run(command, check=True, env=os.environ.copy())
assert (output / 'comparison.json').is_file()
print('Both crop trials finished; inspect validation comparison.json.', flush=True)
""")
    cell("markdown", """Each arm saves best.pt, final.pt, resume.pt, config.json and history.json.
best.pt keeps the source unless validation foreground micro Dice at threshold 0.5
improves. comparison.json audits source and the selected checkpoints at thresholds
0.3-0.9 on the same 562 full validation images: 282 normal and 280 positive.
It reports micro Dice, IoU, precision, recall, normal-image false positives,
and small (<256 GT pixels) versus other positive groups with micro and mean-image
Dice. Choose checkpoint and threshold using validation only. Compare both arms
at the same thresholds as well as their calibrated best. Training Dice is measured
on different augmented views, so it is not the A/B outcome. The validation winner
may remain source. The local demo checkpoint is not changed by this job.
""")
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    output = ROOT / "notebooks/kaggle_btxrd_vit_lesion_crop_ab"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": "vtan2405/xla-btxrd-vit-lesion-crop-ab-20-epochs",
        "title": "XLA BTXRD ViT Lesion Crop AB 20 Epochs", "code_file": "vit_lesion_crop_ab.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": ["vtan2405/xla-btxrd-vit-improvement-ab-20-epochs"],
        "model_sources": [],
    }
    (output / "vit_lesion_crop_ab.ipynb").write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Packaged {len(SOURCE_PATHS)} sources into {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
