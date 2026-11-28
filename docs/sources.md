# Nguồn dữ liệu, mô hình và tài liệu

| Thành phần | Nguồn được dùng trong dự án |
| --- | --- |
| BTXRD | [Bài báo và dữ liệu gốc](https://www.nature.com/articles/s41597-024-04311-y) · [Bản đóng gói dùng trong project](https://huggingface.co/datasets/torusama/xla-seminar-uia) |
| FracAtlas | [Bài báo và dữ liệu gốc](https://www.nature.com/articles/s41597-023-02432-4) · [Cấu hình dự phòng](../configs/datasets/fracatlas.yaml) |
| Vision Transformer | [Bài báo](https://arxiv.org/abs/2010.11929) · [Mã Google JAX/Flax](https://github.com/google-research/vision_transformer) · [Bản PyTorch của timm, ImageNet-21K](https://huggingface.co/timm/vit_base_patch16_224.orig_in21k) |
| Segmenter | [Bài báo](https://openaccess.thecvf.com/content/ICCV2021/html/Strudel_Segmenter_Transformer_for_Semantic_Segmentation_ICCV_2021_paper.html) · [Mã nguồn](https://github.com/rstrudel/segmenter/tree/20d1bfad354165ee45c3f65972a4d9c131f58d53) · [Giấy phép MIT đi kèm mã](../src/xla/vendor/segmenter/LICENSE) |
| EfficientViT | [Bài báo](https://openaccess.thecvf.com/content/ICCV2023/html/Cai_EfficientViT_Lightweight_Multi-Scale_Attention_for_High-Resolution_Dense_Prediction_ICCV_2023_paper.html) · [Mã nguồn MIT HAN Lab, Apache-2.0](https://github.com/mit-han-lab/efficientvit/tree/de7d7733cc0329f391b33f1f459271562ec27bd5) |

Thư mục [paper/](../paper/) chứa bản PDF dùng trong seminar. Mã [src/xla/vendor/segmenter/](../src/xla/vendor/segmenter/) chứa decoder, encoder và các block từ Segmenter tại commit đã ghim. Các adapter chuyển đầu ra thành mask nhị phân cho BTXRD. Bản ViT có skip tích chập khởi tạo encoder từ `torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1`.

Launcher cài EfficientViT từ commit trên với `--no-deps`. Adapter nạp các thành phần segmentation B1, dùng batch normalization và chạy được trên Windows mà không cần Triton RMSNorm. Checkpoint EfficientViT trên Kaggle có SHA-256 `3fd14076d7c0935992556ea7a33f772ce02398c25efa431c3c11acc9ac78e5ae`, khớp với file từ job train đã dùng để báo cáo số liệu. [Gói checkpoint public](https://www.kaggle.com/datasets/vtan2405/bonescope-btxrd-checkpoints) · [Job EfficientViT-Seg](https://www.kaggle.com/code/qunthy/efficientvit-seg)
