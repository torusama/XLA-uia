"""Offline test of threshold counts and normal-image false positives."""
import pytest
import torch

from xla.engine.segmentation_audit import ThresholdAudit


def test_audit_counts_at_multiple_thresholds():
    audit = ThresholdAudit([0.5, 0.8])
    probabilities = torch.tensor([[[[0.9, 0.6]]], [[[0.7, 0.1]]]])
    masks = torch.tensor([[[[1, 0]]], [[[0, 0]]]])
    audit.update(probabilities, masks)
    low, high = audit.results()
    assert (low["tp"], low["fp"], low["fn"]) == (1, 2, 0)
    assert low["normal_images_with_prediction"] == 1
    assert (high["tp"], high["fp"], high["fn"]) == (1, 0, 0)
    assert high["normal_images_with_prediction"] == 0
    assert audit.normal_images == audit.positive_images == 1


def test_audit_rejects_bad_shape():
    audit = ThresholdAudit([0.5])
    with pytest.raises(ValueError, match="same shape"):
        audit.update(torch.zeros(1, 2), torch.zeros(1, 3))
