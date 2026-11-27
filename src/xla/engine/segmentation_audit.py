"""Aggregate validation metrics at several probability thresholds."""
class ThresholdAudit:
    """Keep pixel counts and image-level false positives for each threshold."""

    def __init__(self, thresholds):
        self.thresholds = [float(t) for t in thresholds]
        if not self.thresholds or any(not 0 < t < 1 for t in self.thresholds):
            raise ValueError("Thresholds must be between zero and one")
        self.counts = [[0, 0, 0, 0] for _ in self.thresholds]
        self.normal_images = 0
        self.positive_images = 0

    def update(self, probabilities, target):
        truth = target.bool().flatten(1)
        probabilities = probabilities.flatten(1)
        if probabilities.shape != truth.shape:
            raise ValueError("Probabilities and masks must have the same shape")
        normal = ~truth.any(dim=1)
        self.normal_images += int(normal.sum().item())
        self.positive_images += int((~normal).sum().item())
        for threshold, counts in zip(self.thresholds, self.counts):
            pred = probabilities >= threshold
            counts[0] += int((pred & truth).sum().item())
            counts[1] += int((pred & ~truth).sum().item())
            counts[2] += int((~pred & truth).sum().item())
            counts[3] += int((pred.any(dim=1) & normal).sum().item())

    def results(self):
        results = []
        for threshold, (tp, fp, fn, normal_fp) in zip(self.thresholds, self.counts):
            results.append({"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
                            "dice": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
                            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
                            "precision": tp / (tp + fp) if tp + fp else None,
                            "recall": tp / (tp + fn) if tp + fn else None,
                            "normal_images_with_prediction": normal_fp})
        return results
