"""Aggregate validation metrics at several probability thresholds."""
class ThresholdAudit:
    """Keep pixel counts and image-level false positives for each threshold."""

    def __init__(self, thresholds):
        self.thresholds = [float(t) for t in thresholds]
        if not self.thresholds or any(not 0 < t < 1 for t in self.thresholds):
            raise ValueError("Thresholds must be between zero and one")
        self.counts = [[0, 0, 0, 0] for _ in self.thresholds]
        self.groups = [{name: [0, 0, 0, 0, 0.0] for name in
                        ("small_under_256px", "other_positive")} for _ in self.thresholds]
        self.normal_images = 0
        self.positive_images = 0

    def update(self, probabilities, target):
        truth = target.bool().flatten(1)
        probabilities = probabilities.flatten(1)
        if probabilities.shape != truth.shape:
            raise ValueError("Probabilities and masks must have the same shape")
        normal = ~truth.any(dim=1)
        areas = truth.sum(dim=1)
        self.normal_images += int(normal.sum().item())
        self.positive_images += int((~normal).sum().item())
        for threshold, counts, groups in zip(self.thresholds, self.counts, self.groups):
            pred = probabilities >= threshold
            counts[0] += int((pred & truth).sum().item())
            counts[1] += int((pred & ~truth).sum().item())
            counts[2] += int((~pred & truth).sum().item())
            counts[3] += int((pred.any(dim=1) & normal).sum().item())
            tp = (pred & truth).sum(dim=1)
            fp = (pred & ~truth).sum(dim=1)
            fn = (~pred & truth).sum(dim=1)
            per_image_dice = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
            for name, selected in (("small_under_256px", (areas > 0) & (areas < 256)),
                                   ("other_positive", areas >= 256)):
                values = groups[name]
                for index, metric in enumerate((tp, fp, fn)):
                    values[index] += int(metric[selected].sum().item())
                values[3] += int(selected.sum().item())
                values[4] += float(per_image_dice[selected].sum().item())

    def results(self):
        results = []
        for threshold, (tp, fp, fn, normal_fp), groups in zip(
                self.thresholds, self.counts, self.groups):
            sizes = {}
            for name, (gtp, gfp, gfn, count, dice_sum) in groups.items():
                sizes[name] = {"images": count, "tp": gtp, "fp": gfp, "fn": gfn,
                               "micro_dice": 2 * gtp / (2 * gtp + gfp + gfn) if count else None,
                               "mean_image_dice": dice_sum / count if count else None}
            results.append({"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
                            "dice": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
                            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
                            "precision": tp / (tp + fp) if tp + fp else None,
                            "recall": tp / (tp + fn) if tp + fn else None,
                            "normal_images_with_prediction": normal_fp, "positive_size_groups": sizes})
        return results
