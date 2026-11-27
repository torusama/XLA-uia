"""Paired mild augmentation for X-ray segmentation training only."""
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class PairedAugmentation(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]
        if random.random() < 0.5:
            image, mask = image.flip(-1), mask.flip(-1)
        angle = random.uniform(-7, 7)
        translation = [random.randint(-4, 4), random.randint(-4, 4)]
        scale = random.uniform(0.97, 1.03)
        image = TF.affine(image, angle, translation, scale, [0, 0],
                          interpolation=InterpolationMode.BILINEAR, fill=-1)
        mask = TF.affine(mask, angle, translation, scale, [0, 0],
                         interpolation=InterpolationMode.NEAREST, fill=0)
        return image, mask.to(torch.float32)
