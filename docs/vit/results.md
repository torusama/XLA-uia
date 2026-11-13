# Tr?ng th?i

- PASS: Python 3.11.9, torch 2.15.0+cpu, torchvision 0.29.2+cpu, timm 1.0.31 tr?n Windows; CUDA kh?ng c? trong m?i tr??ng CPU hi?n t?i.
- PASS: `python -m pytest -q`: 8 tests, g?m ViT random-init CPU forward v? gradient head; `python -m ruff check .`; `python -m xla doctor`.
- PASS: `build_vit(pretrained=True)` t?i v? n?p backbone; head Linear(768, 2) kh?i t?o m?i. Checkpoint Hugging Face revision `ba93fb27cf66db1117ac67bb6e9e4a599df98b87`; cache model.safetensors 343208552 bytes. Preprocess t? timm: input 3?224?224, bicubic, mean/std 0.5, crop_pct 0.9. Ch?nh s?ch resize/crop cho X-quang s? ???c quy?t ??nh khi l?m pipeline.
- Ch?a c? FracAtlas: data audit, training, splits v? ??nh gi? th?c t? ch?a th?c hi?n. Test random initialization ch? ki?m tra k? thu?t, kh?ng ph?i k?t qu? th? nghi?m.
