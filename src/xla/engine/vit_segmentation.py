"""Training and evaluation utilities for binary ViT segmentation."""
import torch
from torch.nn import functional as F


def segmentation_loss(logits, target, positive_weight=10.0):
    probabilities = logits.sigmoid()
    if positive_weight is None:
        intersection = (probabilities * target).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        return F.binary_cross_entropy_with_logits(logits, target) + (
            1 - ((2 * intersection + 1) / (denominator + 1)).mean()
        )
    intersection = (probabilities * target).sum()
    denominator = probabilities.sum() + target.sum()
    dice_loss = 1 - (2 * intersection + 1) / (denominator + 1)
    weight = logits.new_tensor(positive_weight)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight) + dice_loss


def run_epoch(model, loader, device, optimizer=None, positive_weight=10.0):
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
            loss = segmentation_loss(logits, target, positive_weight)
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
