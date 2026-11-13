"""CPU-only checks; random initialization is not an experiment result."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from xla.models.registry import build_model
from xla.models.vit import MODEL_NAME, build_vit, preprocessing_config


def test_vit_forward_and_gradient():
    torch.manual_seed(42)
    model = build_vit(pretrained=False)
    assert model.patch_embed.patch_size == (16, 16)
    assert model.embed_dim == 768
    assert len(model.blocks) == 12
    assert model.blocks[0].attn.num_heads == 12
    assert tuple(model.pos_embed.shape) == (1, 197, 768)
    assert model.head.out_features == 2
    assert preprocessing_config(model)["input_size"] == (3, 224, 224)
    model.eval()
    # Head-only gradient avoids a costly full-backbone CPU training step.
    with torch.no_grad():
        features = model.forward_features(torch.zeros(1, 3, 224, 224))
    logits = model.forward_head(features)
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
    assert torch.isfinite(model.head.weight.grad).all()


def test_registry_rejects_unimplemented_model():
    with pytest.raises(ValueError, match="Unsupported"):
        build_model({"name": "efficientvit_b0", "task": "classification"})


def test_pretrained_request_is_forwarded(monkeypatch):
    import timm

    seen = {}
    def fake_create(name, **kwargs):
        seen.update(name=name, **kwargs)
        return object()
    monkeypatch.setattr(timm, "create_model", fake_create)
    build_vit(pretrained=True)
    assert seen == {"name": MODEL_NAME, "pretrained": True, "num_classes": 2}
