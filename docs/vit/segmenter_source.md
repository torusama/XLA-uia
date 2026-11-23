# Nguồn và cấu hình Segmenter

- Paper: Strudel và cộng sự, *Segmenter: Transformer for Semantic Segmentation*, ICCV 2021.
- Mã nguồn chính thức: https://github.com/rstrudel/segmenter, commit `20d1bfad354165ee45c3f65972a4d9c131f58d53`, giấy phép MIT (Robin Strudel và INRIA); bản license lưu cạnh mã trong `src/xla/vendor/segmenter/`.
- Decoder `MaskTransformer` và các block liên quan được lấy từ mã nguồn trên. Import được điều chỉnh cho package dự án và API `timm` hiện tại; phần decoder tuyến tính không được sử dụng. Block gốc ghi nhận dựa trên `timm` của Ross Wightman.
- Backbone là `timm` ViT-B/16 `vit_base_patch16_224.orig_in21k` pretrained ImageNet-21k, Apache-2.0 theo model card https://huggingface.co/timm/vit_base_patch16_224.orig_in21k. Đây không phải checkpoint Segmenter đã train trên ADE20K: decoder và hai class embedding được khởi tạo mới.
- Decoder dùng 2 lớp, 12 head, chiều embedding 768, MLP 3072, dropout 0.1, 2 class token nền/u, theo cấu hình Mask Transformer của source. Đầu ra hai lớp được biểu diễn bằng hiệu logit foreground và background để tái sử dụng pipeline nhị phân.
- Thí nghiệm dùng ảnh 224×224 BTXRD và 50 epoch, giữ split của baseline. Loss BCE có trọng số foreground 10 cộng soft Dice và AdamW giống baseline trước, không phải loss cross-entropy và SGD mặc định của paper. Cấu hình này là thử nghiệm, không phải tái lập đầy đủ kết quả paper.

Notebook chạy Kaggle nằm trong `notebooks/kaggle_btxrd_segmenter/`; job độc lập với baseline cũ. `best.pt` được chọn theo Dice foreground trên validation, `final.pt` ở epoch cuối. Test không tham gia lựa chọn checkpoint.
