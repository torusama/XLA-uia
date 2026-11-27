"""Per-pixel cross-entropy and measurements for the upstream Segmenter."""
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


class RandomHorizontalFlip(Dataset):
    """Apply the same horizontal flip to images and binary masks."""

    def __init__(self, dataset, probability=0.5):
        self.dataset = dataset
        self.probability = probability

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]
        if torch.rand(()) < self.probability:
            image, mask = image.flip(-1), mask.flip(-1)
        return image, mask


def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = count = tp = fp = fn = 0
    with torch.set_grad_enabled(training):
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            if training:
                optimizer.zero_grad()
            scores = model(images)
            loss = F.cross_entropy(scores, target[:, 0].long(), ignore_index=255)
            if training:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            count += images.shape[0]
            total_loss += loss.item() * images.shape[0]
            prediction = scores.detach().argmax(dim=1).bool()
            truth = target[:, 0].bool()
            tp += (prediction & truth).sum().item()
            fp += (prediction & ~truth).sum().item()
            fn += (~prediction & truth).sum().item()
    return {"loss": total_loss / count,
            "dice": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0}
