"""Check crop geometry, full-image controls, and mask integrity."""
import random

import pytest
import torch

from xla.data.lesion_crop import LesionCropDataset


@pytest.mark.parametrize("top,left", [(0, 0), (0, 217), (217, 0), (217, 217), (99, 101)])
def test_crop_keeps_border_lesions_and_paired_geometry(top, left):
    mask = torch.zeros(1, 224, 224)
    mask[:, top:top + 7, left:left + 7] = 1
    image = mask.expand(3, -1, -1) * 2 - 1
    original = image.clone(), mask.clone()
    dataset = LesionCropDataset([(image, mask)], probability=1,
                                min_crop_size=112, max_crop_size=112)
    random.seed(42)
    result_image, result_mask = dataset[0]
    random.seed(42)
    repeated_image, repeated_mask = dataset[0]
    assert torch.equal(result_image, repeated_image)
    assert torch.equal(result_mask, repeated_mask)
    assert result_image.shape == image.shape and result_mask.shape == mask.shape
    assert result_mask.dtype == mask.dtype
    assert set(result_mask.unique().tolist()) == {0, 1}
    assert result_mask.sum() == mask.sum() * 4
    assert ((result_image[:1] > 0) == result_mask.bool()).float().mean() > 0.995
    assert torch.equal(image, original[0]) and torch.equal(mask, original[1])


@pytest.mark.parametrize("kind", ["normal", "large", "dispersed", "disabled", "skip"])
def test_ineligible_or_uncropped_samples_stay_unchanged(kind, monkeypatch):
    image = torch.rand(3, 224, 224)
    mask = torch.zeros(1, 224, 224)
    if kind == "large":
        mask[:, :16, :16] = 1  # Exactly 256 pixels is outside the small group.
    elif kind == "dispersed":
        mask[:, 0, 0] = mask[:, -1, -1] = 1  # Full mask cannot fit in a crop.
    elif kind in ("disabled", "skip"):
        mask[:, 100:105, 100:105] = 1
    if kind == "skip":
        monkeypatch.setattr(random, "random", lambda: 0.75)
    dataset = LesionCropDataset([(image, mask)], probability=0 if kind == "disabled" else 0.5)
    result = dataset[0]
    assert result[0] is image and result[1] is mask
    assert len(dataset) == 1


def test_mixture_keeps_full_image_and_cropped_views():
    mask = torch.zeros(1, 224, 224)
    mask[:, 100:105, 100:105] = 1
    image = mask.expand(3, -1, -1) * 2 - 1
    dataset = LesionCropDataset([(image, mask)], probability=0.5)
    random.seed(42)
    areas = [int(dataset[0][1].sum()) for _ in range(40)]
    assert any(area == 25 for area in areas) and any(area > 25 for area in areas)


@pytest.mark.parametrize("kwargs", [{"probability": -0.1}, {"probability": 1.1},
                                    {"min_crop_size": 0}, {"max_crop_size": 100},
                                    {"max_lesion_pixels": 1}, {"context": -1}])
def test_invalid_crop_configuration(kwargs):
    with pytest.raises(ValueError):
        LesionCropDataset([], **kwargs)
