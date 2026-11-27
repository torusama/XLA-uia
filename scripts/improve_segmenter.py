"""Compare controlled Segmenter fine-tuning arms on validation only."""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from xla.data.augmentation import PairedAugmentation
from xla.data.hf_segmentation import make_splits
from xla.engine.segmentation_audit import ThresholdAudit
from xla.engine.vit_segmentation import run_epoch
from xla.models.segmenter_vit import SegmenterViT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(model, loader, device):
    result = ThresholdAudit([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    model.eval()
    with torch.inference_mode():
        for image, mask in loader:
            result.update(model(image.to(device)).sigmoid(), mask.to(device))
    return {"thresholds": result.results(),
            "best": max(result.results(), key=lambda value: value["dice"]),
            "normal_images": result.normal_images, "positive_images": result.positive_images}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--weights", type=float, nargs="+", default=[50.0, 10.0])
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.learning_rate <= 0
            or any(weight <= 0 for weight in args.weights) or len(set(args.weights)) != len(args.weights)):
        parser.error("Invalid training parameters")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    digest = sha256(args.archive)
    if source.get("archive_sha256") != digest:
        raise ValueError("Checkpoint and dataset checksum do not match")
    old = source["settings"]
    if old.get("architecture") != "segmenter_vit_mask_transformer":
        raise ValueError("Expected the Segmenter ViT checkpoint")
    train, val, _ = make_splits(args.archive)
    print(f"Training={len(train)}, validation={len(val)}; test is not evaluated", flush=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"split": "validation", "archive_sha256": digest,
               "source_checkpoint_sha256": sha256(args.checkpoint), "arms": {}}
    for weight in args.weights:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = SegmenterViT(pretrained=False, mask_norm_eps=old["mask_norm_eps"])
        model.load_state_dict(source["model"], strict=True)
        model.to(args.device)
        if "source" not in summary:
            summary["source"] = audit(model, val_loader, args.device)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        arm = args.output / f"weight_{weight:g}"
        arm.mkdir()
        settings = dict(old, positive_weight=weight, learning_rate=args.learning_rate,
                        epochs=args.epochs, seed=args.seed, batch_size=args.batch_size,
                        paired_augmentation=True, optimizer_state="reinitialized",
                        source_checkpoint_sha256=summary["source_checkpoint_sha256"])
        (arm / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        loader = DataLoader(PairedAugmentation(train), batch_size=args.batch_size,
                            shuffle=True, num_workers=args.workers)
        best = next(x["dice"] for x in summary["source"]["thresholds"] if x["threshold"] == 0.5)
        shutil.copyfile(args.checkpoint, arm / "best.pt")
        history = []
        for epoch in range(1, args.epochs + 1):
            training = run_epoch(model, loader, args.device, optimizer, weight)
            validation = run_epoch(model, val_loader, args.device, positive_weight=weight)
            record = {"epoch": epoch, "positive_weight": weight, "train": training, "val": validation}
            history.append(record)
            print(json.dumps(record), flush=True)
            (arm / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            checkpoint = {"model": model.state_dict(), "epoch": epoch, "val": validation,
                          "archive_sha256": digest, "settings": settings}
            if validation["dice"] > best:
                best = validation["dice"]
                torch.save(checkpoint, arm / "best.pt")
            if epoch % 5 == 0 or epoch == args.epochs:
                torch.save(dict(checkpoint, optimizer=optimizer.state_dict(),
                                torch_rng_state=torch.get_rng_state(),
                                python_rng_state=random.getstate(),
                                cuda_rng_states=torch.cuda.get_rng_state_all()
                                if torch.cuda.is_available() else []), arm / "resume.pt")
            if epoch == args.epochs:
                torch.save(checkpoint, arm / "final.pt")
        del optimizer, loader, checkpoint
        winner = torch.load(arm / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(winner["model"])
        summary["arms"][f"weight_{weight:g}"] = audit(model, val_loader, args.device)
        del winner, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        (args.output / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    candidates = {"source": summary["source"], **summary["arms"]}
    summary["validation_winner"] = max(candidates, key=lambda name: candidates[name]["best"]["dice"])
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"validation_winner": summary["validation_winner"]}), flush=True)


if __name__ == "__main__":
    main()
