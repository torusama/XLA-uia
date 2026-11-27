"""Segmenter built from the authors' ViT and Mask Transformer modules."""
from torch import nn
from torch.nn import functional as F

from xla.vendor.segmenter.decoder import MaskTransformer
from xla.vendor.segmenter.vit import VisionTransformer

BACKBONE = "vit_base_patch16_384.augreg_in21k_ft_in1k"


class OfficialSegmenter(nn.Module):
    """Two-class Segmenter with the upstream encoder and decoder."""

    def __init__(self, pretrained=True, image_size=224):
        super().__init__()
        self.encoder = VisionTransformer(
            image_size=(image_size, image_size), patch_size=16,
            n_layers=12, d_model=768, d_ff=3072, n_heads=12,
            n_cls=1000, dropout=0.0, drop_path_rate=0.1,
        )
        if pretrained:
            import timm

            source = timm.create_model(BACKBONE, pretrained=True, img_size=image_size)
            self.encoder.load_state_dict(source.state_dict(), strict=True)
            del source
        self.decoder = MaskTransformer(
            n_cls=2, patch_size=16, d_encoder=768, n_layers=2,
            n_heads=12, d_model=768, d_ff=3072, drop_path_rate=0.0, dropout=0.1,
        )

    def forward(self, images):
        tokens = self.encoder(images, return_features=True)[:, 1:]
        masks = self.decoder(tokens, images.shape[-2:])
        return F.interpolate(masks, size=images.shape[-2:],
                             mode="bilinear", align_corners=False)
