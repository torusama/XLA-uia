"""Training-only paired crops that magnify small lesions without dropping their mask."""
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class LesionCropDataset(Dataset):
    """Keep every sample; optionally crop small positive masks with surrounding context.

    Wrap paired geometric augmentation so the crop retains its entire resulting
    foreground. Empty masks and larger lesions always keep the full image.
    """

    def __init__(self, dataset, probability=0.5, max_lesion_pixels=256,
                 min_crop_size=112, max_crop_size=196, context=8):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be between 0 and 1")
        if (max_lesion_pixels < 2 or min_crop_size < 1
                or max_crop_size < min_crop_size or context < 0):
            raise ValueError("Invalid lesion crop dimensions")
        self.dataset = dataset
        self.config = {"probability": probability, "max_lesion_pixels": max_lesion_pixels,
                       "min_crop_size": min_crop_size, "max_crop_size": max_crop_size,
                       "context": context, "after_paired_geometry": True}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]
        probability = self.config["probability"]
        if probability == 0:
            return image, mask
        positions = torch.nonzero(mask[0] > 0, as_tuple=False)
        if not 0 < len(positions) < self.config["max_lesion_pixels"]:
            return image, mask
        if random.random() >= probability:
            return image, mask
        height, width = mask.shape[-2:]
        context = self.config["context"]
        y0, x0 = (max(0, int(value) - context) for value in positions.amin(dim=0))
        y1, x1 = (min(limit, int(value) + 1 + context)
                  for value, limit in zip(positions.amax(dim=0), (height, width)))
        minimum = max(self.config["min_crop_size"], y1 - y0, x1 - x0)
        if minimum >= min(height, width):
            return image, mask
        maximum = min(min(height, width), max(minimum, self.config["max_crop_size"]))
        side = random.randint(minimum, maximum)
        top = random.randint(max(0, y1 - side), min(y0, height - side))
        left = random.randint(max(0, x1 - side), min(x0, width - side))
        image = TF.resized_crop(image, top, left, side, side, [height, width],
                                interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resized_crop(mask, top, left, side, side, [height, width],
                               interpolation=InterpolationMode.NEAREST)
        return image, mask
