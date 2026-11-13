# Danh m?c ngu?n

C?c link theo ??c t?; license, commit/version v? t?nh ch?nh th?c ch?a ???c x?c minh ??c l?p: **TODO_VERIFY**. Kh?ng vendor code.

| Ch? ?? | Paper | Source ???c d?n | Task | License / commit |
|---|---|---|---|---|
| ViT | [Paper](https://arxiv.org/abs/2010.11929) | [Source](https://github.com/google-research/vision_transformer) | TODO_VERIFY | TODO_VERIFY |
| EfficientViT Cai 2023 | [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Cai_EfficientViT_Lightweight_Multi-Scale_Attention_for_High-Resolution_Dense_Prediction_ICCV_2023_paper.html) | [Source](https://github.com/mit-han-lab/efficientvit) | TODO_VERIFY | TODO_VERIFY |
| DeiT | [Paper](https://proceedings.mlr.press/v139/touvron21a.html) | [Source](https://github.com/facebookresearch/deit) | TODO_VERIFY | TODO_VERIFY |
| Swin | [Paper](https://arxiv.org/abs/2103.14030) | [Source](https://github.com/microsoft/Swin-Transformer) | TODO_VERIFY | TODO_VERIFY |
| TransUNet | [Paper](https://arxiv.org/abs/2102.04306) | [Source](https://github.com/Beckschen/TransUNet) | TODO_VERIFY | TODO_VERIFY |
| SegFormer | [Paper](https://arxiv.org/abs/2105.15203) | [Source](https://github.com/NVlabs/SegFormer) | TODO_VERIFY | TODO_VERIFY |
| SegNeXt | [Paper](https://arxiv.org/abs/2209.08575) | [Source](https://github.com/Visual-Attention-Network/SegNeXt) | TODO_VERIFY | TODO_VERIFY |
| TopFormer | [Paper](https://arxiv.org/abs/2204.05525) | [Source](https://github.com/hustvl/TopFormer) | TODO_VERIFY | TODO_VERIFY |
| SeaFormer | [Paper](https://openreview.net/forum?id=-qg8MQNrxZw) | [Source](https://github.com/fudan-zvg/SeaFormer) | TODO_VERIFY | TODO_VERIFY |
| FracAtlas | [Paper](https://www.nature.com/articles/s41597-023-02432-4) | [Source](https://github.com/XLR8-07/FracAtlas) | TODO_VERIFY | TODO_VERIFY |
| BTXRD | [Paper](https://www.nature.com/articles/s41597-024-04311-y) | [Source](https://github.com/SHUNHANYAO/BTXRD) | TODO_VERIFY | TODO_VERIFY |

Framework adapter: [timm](https://github.com/huggingface/pytorch-image-models), [Transformers](https://huggingface.co/docs/transformers/tasks/image_classification). EfficientViT: [classification](https://github.com/mit-han-lab/efficientvit/tree/master/applications/efficientvit_cls), [segmentation](https://github.com/mit-han-lab/efficientvit/tree/master/applications/efficientvit_seg). Checkpoint segmentation t?ng qu?t c?n adaptation/fine-tune tr??c khi d?ng cho fracture.

ViT PyTorch port: [timm model card](https://huggingface.co/timm/vit_base_patch16_224.orig_in21k), Apache-2.0 theo model card, revision weights `ba93fb27cf66db1117ac67bb6e9e4a599df98b87`; timm 1.0.31 trong m?i tr??ng ?? ki?m tra. Code ViT g?c c?a Google l? JAX/Flax; timm l? PyTorch port.
