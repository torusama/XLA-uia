"""Package a ViT run using the BTXRD loader, augmentation and scorer from EViT."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "notebooks/kaggle_efficientvit_seg/efficientvit_seg_btxrd.ipynb"
OUTPUT = ROOT / "notebooks/kaggle_btxrd_vit_detail"


def build():
    reference = json.loads(REFERENCE.read_text(encoding="utf8"))
    sources = ["".join(cell["source"]) for cell in reference["cells"]]
    assignment = next(node for node in ast.parse(sources[3]).body if isinstance(node, ast.Assign)
                      and getattr(node.targets[0], "id", "") == "SOURCES")
    loader = ast.literal_eval(assignment.value)["src/xla/data/hf_segmentation.py"]
    augment = next(node for node in ast.parse(sources[4]).body if isinstance(node, ast.ClassDef)
                   and node.name == "TrainAugmentDataset")
    _, marker, evaluation = sources[5].partition("def predict_tta(images):")
    if not marker:
        raise ValueError("Missing EViT test-time ensemble and scorer")
    evaluation = (marker + evaluation).replace("evit_seg_test_metrics.json", "vit_detail_test_metrics.json")
    cells = []

    def cell(kind, source):
        data = {"cell_type": kind, "metadata": {}, "id": f"vit-detail-{len(cells)}",
                "source": source.strip().splitlines(keepends=True)}
        if kind == "code":
            compile(source.strip(), data["id"], "exec")
            data.update(outputs=[], execution_count=None)
        cells.append(data)

    cell("markdown", """# ViT-B/16 with detail skips on the EfficientViT BTXRD split

The convolutional decoder fuses the 14×14 ViT features with image features at
28×28, 56×56 and 112×112. Initialize the encoder from ImageNet-1K only.
Use the same positive training images, normalization, flips, rotations, loss,
50 epochs, batch size and held-out scoring as qunthy/efficientvit-seg.
Select the checkpoint on validation micro Dice at 0.5. Apply the fixed
original/flip ensemble and threshold 0.5 once on test after selection.
""")
    cell("code", """import os
import json
import random
import time
from pathlib import Path

os.environ['HF_HUB_CACHE'] = '/kaggle/temp/huggingface/hub'
import numpy as np
import torch
import torchvision
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

torch.hub.set_dir('/kaggle/temp/torch')
assert torch.cuda.is_available(), 'Kaggle GPU required'
device = torch.device('cuda')
random.seed(42)
torch.manual_seed(42)
print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(), flush=True)
""")
    cell("code", sources[2])
    cell("code", loader)
    cell("code", ast.get_source_segment(sources[4], augment))
    cell("code", """train_set, val_set, test_set = make_splits(archive)
train_dataset = TrainAugmentDataset(train_set)
samples = {
    'train': [train_set.samples[i] for i in train_dataset.indices],
    'val': val_set.samples,
    'test': test_set.samples,
}
expected = {
    'train': (1302, 'd0c6d06e7bec9f37e2dcbdd18231e028b16759231e8cb6c654016d51f58f5db6'),
    'val': (562, '2b2f1d62214c72f4cc7ca8c8bf668bc7153ff7f92b1889f90c1124c187b18a61'),
    'test': (561, 'e151e8391d3d7025b674a0b5d8faf832362f44c7e446c9d526dd9f85305a4c8f'),
}
manifest = {}
for split, rows in samples.items():
    content = ''.join('\\t'.join(row) + '\\n' for row in rows)
    signature = hashlib.sha256(content.encode()).hexdigest()
    assert (len(rows), signature) == expected[split], f'{split} membership differs from EViT'
    manifest[split] = {'count': len(rows), 'sha256': signature}
print('Verified EViT split:', manifest, flush=True)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=8, num_workers=2)
test_loader = DataLoader(test_set, batch_size=8, num_workers=2)
""")
    cell("code", (ROOT / "src/xla/models/vit_detail.py").read_text(encoding="utf8"))
    cell("code", (ROOT / "src/xla/engine/vit_segmentation.py").read_text(encoding="utf8"))
    cell("code", """output = Path('/kaggle/working')
model = DetailViT(pretrained=True).to(device)
head = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
optimizer = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': 2e-5},
                               {'params': head, 'lr': 2e-4}], weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
checkpoint_path = output / 'best_vit_detail.pt'
history = []
best = -1.0
for epoch in range(1, 51):
    train = run_epoch(model, train_loader, device, optimizer, positive_weight=10)
    val = run_epoch(model, val_loader, device, positive_weight=10)
    history.append({'epoch': epoch, 'train': train, 'val': val, 'lr': scheduler.get_last_lr()})
    print(json.dumps(history[-1]), flush=True)
    if val['dice'] > best:
        best = val['dice']
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'validation': val,
                    'archive_sha256': digest, 'split_manifest': manifest,
                    'architecture': 'DetailViT', 'pretraining': 'ViT_B_16_Weights.IMAGENET1K_V1'},
                   checkpoint_path)
    scheduler.step()
    (output / 'history.json').write_text(json.dumps(history, indent=2))
""")
    cell("code", """checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()
""" + evaluation)
    cell("code", """# Published best-checkpoint test output of qunthy/efficientvit-seg
evit = {'global_micro_dice': 0.7403773162089105, 'global_micro_iou': 0.5877770587463502,
        'mean_macro_dice_tumor': 0.5708889301999, 'mean_macro_iou_tumor': 0.46695038501914377,
        'specificity_normal': 0.6903914590747331}
assert (metrics['tumor_test_images'], metrics['normal_test_images']) == (280, 281)
comparison = {'reference': 'qunthy/efficientvit-seg', 'archive_sha256': digest,
              'split_manifest': manifest, 'selected_epoch': checkpoint['epoch'],
              'validation_dice': checkpoint['validation']['dice'], 'threshold': 0.5,
              'test_tta': 'mean of original and unflipped horizontal-flip probabilities',
              'evit_test': evit, 'vit_detail_test': metrics,
              'vit_minus_evit_test': {key: metrics[key] - value for key, value in evit.items()}}
(output / 'comparison.json').write_text(json.dumps(comparison, indent=2))
print(json.dumps(comparison, indent=2))
""")
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "vit_detail.ipynb").write_text(json.dumps(build(), indent=1) + "\n", encoding="utf8")
