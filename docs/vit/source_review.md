# Ngu?n ViT

[Paper g?c](https://arxiv.org/abs/2010.11929) m? t? bi?n ?nh th?nh chu?i patch v? h?c attention tr?n c?c token. [Google models_vit.py](https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py) l? implementation JAX/Flax c?a t?c gi?. [timm vision_transformer.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py) l? implementation PyTorch ???c t?i s? d?ng trong adapter; kh?ng ph?i code PyTorch do t?c gi? ViT vi?t.

[Model card orig_in21k](https://huggingface.co/timm/vit_base_patch16_224.orig_in21k) ghi backbone pretrain tr?n ImageNet-21k b?ng JAX, ???c Ross Wightman port sang PyTorch, kh?ng c? classification head; head hai l?p kh?i t?o m?i. Model card ghi Apache-2.0. Revision checkpoint ?? t?i: `ba93fb27cf66db1117ac67bb6e9e4a599df98b87`; checksum TODO_VERIFY. Phi?n b?n th? vi?n trong m?i tr??ng ???c ghi ? results.md. Kh?ng vendor code b?n th? ba.
