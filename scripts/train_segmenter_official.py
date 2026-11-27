"""Fine-tune the authors' Segmenter architecture and training method on BTXRD."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from xla.data.hf_segmentation import make_splits
from xla.engine.segmenter_official import RandomHorizontalFlip, run_epoch
from xla.models.segmenter_official import BACKBONE, OfficialSegmenter


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from-scratch", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers must be nonnegative")
    if args.output.exists():
        raise FileExistsError(f"Output directory already exists: {args.output}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_set, val_set, test_set = make_splits(args.archive)
    print(f"Samples: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}", flush=True)
    train_loader = DataLoader(RandomHorizontalFlip(train_set), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)
    model = OfficialSegmenter(pretrained=not args.from_scratch).to(device)
    base_lr, min_lr, power = 0.001, 0.00001, 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0)
    max_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (min_lr + (base_lr - min_lr) *
                                 max(0.0, 1 - step / max_steps) ** power) / base_lr)
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {"architecture": "segmenter_official_mask_transformer",
                "source_commit": "20d1bfad354165ee45c3f65972a4d9c131f58d53",
                "backbone": BACKBONE, "pretrained": not args.from_scratch,
                "pretrained_decoder": False, "image_size": 224, "classes": 2,
                "loss": "cross_entropy", "optimizer": "SGD", "momentum": 0.9,
                "weight_decay": 0.0, "learning_rate": base_lr, "min_lr": min_lr,
                "scheduler": "polynomial", "poly_power": power,
                "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
                "augmentation": "horizontal_flip", "archive_sha256": sha256(args.archive),
                "excluded_train": train_set.excluded, "excluded_val": val_set.excluded,
                "excluded_test": test_set.excluded}
    (args.output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"Starting official Segmenter training: {args.epochs} epochs, device={device}", flush=True)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, train_loader, device, optimizer, scheduler)
        validation = run_epoch(model, val_loader, device)
        record = {"epoch": epoch, "train": training, "val": validation,
                  "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(record)
        print(json.dumps(record), flush=True)
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        checkpoint = {"model": model.state_dict(), "epoch": epoch, "val": validation,
                      "archive_sha256": settings["archive_sha256"], "settings": settings}
        if validation["dice"] > best:
            best = validation["dice"]
            torch.save(checkpoint, args.output / "best.pt")
        if epoch == args.epochs:
            torch.save(checkpoint, args.output / "final.pt")


if __name__ == "__main__":
    main()
