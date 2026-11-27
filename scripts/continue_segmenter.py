"""Continue a Segmenter run from its saved epoch-50 weights."""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from xla.data.hf_segmentation import make_splits
from xla.engine.vit_segmentation import run_epoch
from xla.models.segmenter_vit import SegmenterViT


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prior-best", type=Path, required=True)
    parser.add_argument("--prior-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.000002)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.additional_epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("additional-epochs and batch-size must be positive; workers nonnegative")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    if args.output.exists():
        raise FileExistsError(f"Output directory already exists: {args.output}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    prior_best = torch.load(args.prior_best, map_location="cpu", weights_only=True)
    history = json.loads(args.prior_history.read_text(encoding="utf-8"))
    start = checkpoint["epoch"]
    if start != 50 or [entry["epoch"] for entry in history] != list(range(1, 51)):
        raise ValueError("Expected the final checkpoint and complete history through epoch 50")
    if checkpoint.get("settings", {}).get("architecture") != "segmenter_vit_mask_transformer":
        raise ValueError("Checkpoint is not the previous Mask Transformer run")
    digest = sha256(args.archive)
    if checkpoint["archive_sha256"] != digest or prior_best["archive_sha256"] != digest:
        raise ValueError("Checkpoint was trained on a different archive")
    previous_winner = max(history, key=lambda record: record["val"]["dice"])
    if prior_best["epoch"] != previous_winner["epoch"]:
        raise ValueError("Prior best checkpoint does not match the previous history")
    if checkpoint["val"] != history[-1]["val"] or prior_best["val"] != previous_winner["val"]:
        raise ValueError("Checkpoint validation metrics do not match the previous history")
    if "optimizer" in checkpoint or "optimizer_state_dict" in checkpoint:
        raise ValueError("Expected the original weights-only checkpoint")
    settings = dict(checkpoint["settings"])
    mask_norm_eps = settings["mask_norm_eps"]
    positive_weight = settings["positive_weight"]
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_set, val_set, test_set = make_splits(args.archive)
    print(f"Samples: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}", flush=True)
    model = SegmenterViT(pretrained=False, mask_norm_eps=mask_norm_eps).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint, prior_best
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)
    settings.update({"start_epoch": start, "epochs": start + args.additional_epochs,
                     "learning_rate": args.learning_rate, "batch_size": args.batch_size,
                     "continuation_seed": args.seed, "optimizer_state": "reinitialized",
                     "source_checkpoint_sha256": sha256(args.checkpoint),
                     "prior_best_epoch": previous_winner["epoch"]})
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.prior_best, args.output / "best.pt")
    (args.output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best = previous_winner["val"]["dice"]
    print(f"Loaded epoch {start}; optimizer initialized anew; previous best "
          f"epoch={previous_winner['epoch']} dice={best:.6f}", flush=True)
    for epoch in range(start + 1, start + args.additional_epochs + 1):
        training = run_epoch(model, train_loader, device, optimizer, positive_weight)
        validation = run_epoch(model, val_loader, device, positive_weight=positive_weight)
        record = {"epoch": epoch, "train": training, "val": validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        output = {"model": model.state_dict(), "epoch": epoch, "val": validation,
                  "archive_sha256": digest, "settings": settings}
        if validation["dice"] > best:
            best = validation["dice"]
            torch.save(output, args.output / "best.pt")
        if epoch % 5 == 0 or epoch == start + args.additional_epochs:
            snapshot = dict(output, optimizer=optimizer.state_dict(),
                            torch_rng_state=torch.get_rng_state(),
                            cuda_rng_states=torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available() else [],
                            python_rng_state=random.getstate())
            torch.save(snapshot, args.output / "resume.pt")
        if epoch == start + args.additional_epochs:
            torch.save(output, args.output / "final.pt")


if __name__ == "__main__":
    main()
