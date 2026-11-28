"""Training and evaluation utilities for binary ViT segmentation."""
import torch
from torch.nn import functional as F


def segmentation_loss(logits, target, positive_weight=10.0, dice_reduction="batch", small_dice_weight=0.0):
    if dice_reduction not in ("batch", "image"):
        raise ValueError("dice_reduction must be batch or image")
    probabilities = logits.sigmoid()
    if positive_weight is None:
        intersection = (probabilities * target).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        return F.binary_cross_entropy_with_logits(logits, target) + (
            1 - ((2 * intersection + 1) / (denominator + 1)).mean()
        )
    dimensions = (1, 2, 3) if dice_reduction == "image" else (0, 1, 2, 3)
    intersection = (probabilities * target).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + target.sum(dim=dimensions)
    dice_loss = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    weight = logits.new_tensor(positive_weight)
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight) + dice_loss
    if small_dice_weight:
        areas = target.sum(dim=(1, 2, 3))
        selected = (areas > 0) & (areas < 256)
        if selected.any():
            small_probs, small_targets = probabilities[selected], target[selected]
            intersection = (small_probs * small_targets).sum(dim=(1, 2, 3))
            denominator = small_probs.sum(dim=(1, 2, 3)) + small_targets.sum(dim=(1, 2, 3))
            loss = loss + small_dice_weight * (1 - (2 * intersection + 1) / (denominator + 1)).mean()
    return loss


def run_epoch(model, loader, device, optimizer=None, positive_weight=10.0,
              dice_reduction="batch", small_dice_weight=0.0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    tp = fp = fn = 0
    count = 0
    with torch.set_grad_enabled(training):
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = segmentation_loss(logits, target, positive_weight, dice_reduction, small_dice_weight)
            if training:
                loss.backward()
                optimizer.step()
            batch = images.shape[0]
            total_loss += loss.item() * batch
            count += batch
            prediction = logits.detach() >= 0
            truth = target.bool()
            tp += (prediction & truth).sum().item()
            fp += (prediction & ~truth).sum().item()
            fn += (~prediction & truth).sum().item()
    return {"loss": total_loss / count, "dice": (2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 1.0,
            "iou": (tp / (tp + fp + fn)) if tp + fp + fn else 1.0}
