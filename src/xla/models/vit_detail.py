"""ViT-B/16 segmentation with convolutional skip features from the input image."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16


def conv_block(inputs, outputs, stride=1):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, outputs),
        nn.GELU(),
    )


def upsample(features, reference):
    return F.interpolate(features, size=reference.shape[-2:], mode="bilinear", align_corners=False)


class DetailViT(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = vit_b_16(weights=weights)
        self.backbone.heads = nn.Identity()
        self.half_detail = conv_block(3, 24, stride=2)
        self.quarter_detail = conv_block(24, 48, stride=2)
        self.eighth_detail = conv_block(48, 64, stride=2)
        self.context = nn.Conv2d(self.backbone.hidden_dim, 128, 1)
        self.decode_eighth = conv_block(128 + 64, 96)
        self.decode_quarter = conv_block(96 + 48, 64)
        self.decode_half = conv_block(64 + 24, 32)
        self.mask = nn.Conv2d(32, 1, 1)

    def forward(self, image):
        patches = self.backbone.conv_proj(image)
        tokens = patches.flatten(2).transpose(1, 2)
        cls = self.backbone.class_token.expand(image.size(0), -1, -1)
        encoded = self.backbone.encoder(torch.cat((cls, tokens), dim=1))[:, 1:]
        context = self.context(encoded.transpose(1, 2).reshape_as(patches))
        half = self.half_detail(image)
        quarter = self.quarter_detail(half)
        eighth = self.eighth_detail(quarter)
        x = self.decode_eighth(torch.cat((upsample(context, eighth), eighth), dim=1))
        x = self.decode_quarter(torch.cat((upsample(x, quarter), quarter), dim=1))
        x = self.decode_half(torch.cat((upsample(x, half), half), dim=1))
        return upsample(self.mask(x), image)
