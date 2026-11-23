"""ViT-B/16 with the original Segmenter Mask Transformer decoder."""
from torch import nn
from torch.nn import functional as F

from xla.models.vit import build_vit
from xla.vendor.segmenter.decoder import MaskTransformer


class SegmenterViT(nn.Module):
    """Binary segmentation using two Segmenter class tokens."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = build_vit(pretrained=pretrained, num_classes=2)
        self.backbone.reset_classifier(0)
        dim = self.backbone.embed_dim
        self.decoder = MaskTransformer(
            n_cls=2, patch_size=16, d_encoder=dim, n_layers=2,
            n_heads=dim // 64, d_model=dim, d_ff=4 * dim,
            drop_path_rate=0.0, dropout=0.1,
        )

    def forward(self, images):
        tokens = self.backbone.forward_features(images)
        tokens = tokens[:, self.backbone.num_prefix_tokens:, :]
        logits = self.decoder(tokens, images.shape[-2:])
        logits = F.interpolate(logits, size=images.shape[-2:],
                               mode="bilinear", align_corners=False)
        # Two-class softmax is equivalent to sigmoid(foreground - background).
        return logits[:, 1:2] - logits[:, 0:1]
