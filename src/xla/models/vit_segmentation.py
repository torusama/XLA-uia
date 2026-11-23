"""A simple dense prediction head on the original ViT-B/16 backbone."""
import torch
from torch import nn
from torch.nn import functional as F

from xla.models.vit import build_vit


class ViTSegmentation(nn.Module):
    """Predict one foreground logit per pixel from ViT patch tokens."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = build_vit(pretrained=pretrained, num_classes=2)
        self.backbone.reset_classifier(0)
        self.head = nn.Sequential(
            nn.Conv2d(self.backbone.embed_dim, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 1, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(images)
        tokens = tokens[:, self.backbone.num_prefix_tokens:, :]
        height, width = self.backbone.patch_embed.grid_size
        features = tokens.transpose(1, 2).reshape(images.shape[0], -1, height, width)
        return F.interpolate(self.head(features), size=images.shape[-2:],
                             mode="bilinear", align_corners=False)
