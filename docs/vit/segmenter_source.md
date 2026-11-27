# Nguồn và cấu hình Segmenter

- Paper: Strudel và cộng sự, *Segmenter: Transformer for Semantic Segmentation*, ICCV 2021.
- Mã nguồn chính thức: https://github.com/rstrudel/segmenter, commit `20d1bfad354165ee45c3f65972a4d9c131f58d53`, giấy phép MIT (Robin Strudel và INRIA); bản license lưu cạnh mã trong `src/xla/vendor/segmenter/`.
- Decoder `MaskTransformer` và các block liên quan được lấy từ mã nguồn trên. Import được điều chỉnh cho package dự án và API `timm` hiện tại; phần decoder tuyến tính không được sử dụng. Block gốc ghi nhận dựa trên `timm` của Ross Wightman.
- Backbone là `timm` ViT-B/16 `vit_base_patch16_224.orig_in21k` pretrained ImageNet-21k, Apache-2.0 theo model card https://huggingface.co/timm/vit_base_patch16_224.orig_in21k. Đây không phải checkpoint Segmenter đã train trên ADE20K: decoder và hai class embedding được khởi tạo mới.
- Decoder dùng 2 lớp, 12 head, chiều embedding 768, MLP 3072, dropout 0.1, 2 class token nền/u, theo cấu hình Mask Transformer của source. Đầu ra hai lớp được biểu diễn bằng hiệu logit foreground và background để tái sử dụng pipeline nhị phân.
- Thí nghiệm dùng ảnh 224×224 BTXRD và 50 epoch, giữ split của baseline. Job đầu dùng loss BCE có trọng số foreground 10 cộng soft Dice và AdamW giống baseline trước, không phải loss cross-entropy và SGD mặc định của paper. Cấu hình này là thử nghiệm, không phải tái lập đầy đủ kết quả paper.

Job version 1 bị kẹt ở dự đoán gần như đồng nhất. Với hai lớp, `LayerNorm(2)` của decoder gốc có thể bão hòa khi chênh lệch logit lớn hơn epsilon. Job sửa dùng `mask_norm_eps=0.01` thay vì `1e-5`, learning rate `1e-5` và trọng số foreground `100`; kiến trúc và hai class token vẫn giữ nguyên. Trên hai ảnh train, thử nghiệm chỉ tối ưu decoder với feature backbone cố định cho Dice 0,196 sau 60 bước; đây không phải kết quả validation hoặc bằng chứng đảm bảo hội tụ. Cần đọc kết quả job mới để đánh giá thực tế. Epsilon được lưu trong checkpoint vì nó không nằm trong `state_dict`; checkpoint cũ mặc định dùng `1e-5`.

Notebook chạy Kaggle nằm trong `notebooks/kaggle_btxrd_segmenter/`; job độc lập với baseline cũ. `best.pt` được chọn theo Dice foreground trên validation, `final.pt` ở epoch cuối. Test không tham gia lựa chọn checkpoint.

## Đối chiếu source và kiểm tra job version 2

- So sánh AST xác nhận lớp `MaskTransformer` và `Block` được dùng trong dự án trùng với lớp tương ứng của commit upstream đã ghi ở trên; thay đổi import và epsilon được ghi riêng.
- [Source train gốc](https://github.com/rstrudel/segmenter/blob/20d1bfad354165ee45c3f65972a4d9c131f58d53/segm/engine.py) dùng cross-entropy theo pixel; [cấu hình train](https://github.com/rstrudel/segmenter/blob/20d1bfad354165ee45c3f65972a4d9c131f58d53/segm/train.py) mặc định dùng SGD và polynomial learning-rate scheduler. Các mặc định này áp dụng cho tập dữ liệu của tác giả, không bảo đảm tối ưu cho BTXRD.
- Version 2 dùng encoder `orig_in21k` của `timm`, decoder khởi tạo mới, BCE trọng số 100 cộng soft Dice, AdamW, epsilon `0.01`, ảnh 224×224 và không có augmentation. Đây không phải nguyên cấu hình upstream hay checkpoint pretrained segmentation của họ.
- Version 2 đạt Dice foreground gộp trên validation cao nhất 0,5014 ở epoch 46. Trên một mẫu thăm dò 16 ảnh validation (8 ảnh bình thường, 8 ảnh có u, chọn ngẫu nhiên seed 42), ngưỡng 0,5 cho TP 23.010, FP 18.945, FN 4.402 pixel; 1 trong 8 ảnh bình thường có dự đoán foreground. Mẫu nhỏ này chỉ gợi ý cần kiểm tra tô dư, không đại diện cho toàn tập.

## Train tiếp epoch 51–100

Notebook `notebooks/kaggle_btxrd_segmenter_continue/` đọc đúng `final.pt` epoch 50 của job version 2, đối chiếu SHA-256, lịch sử 50 epoch và checkpoint tốt nhất epoch 46. Kiến trúc, loss, positive weight 100, epsilon 0,01, batch 4 và split giữ nguyên; learning rate giảm từ 1e-5 xuống 2e-6 để thử giảm dao động. Đây là giả thuyết thử nghiệm, không bảo đảm cải thiện validation.

Checkpoint epoch 50 không có trạng thái optimizer hay random, nên AdamW và random được khởi tạo lại; đây là fine-tune tiếp từ trọng số, không phải khôi phục chính xác trạng thái train ban đầu. `best.pt` của job mới giữ nguyên checkpoint tốt nhất từ cả hai giai đoạn; `final.pt` là epoch 100, `history.json` bao gồm cả 100 epoch. `resume.pt` lưu optimizer và random state mỗi 5 epoch để có thể xây dựng quy trình phục hồi sau này.

## Kiểm tra ngưỡng trên validation

Checkpoint tốt nhất epoch 87 (SHA-256 `6c4846f0da78a4964a0fa08556bb0888d9e2683270398c1e718240c532a06a98`) được đo trên toàn bộ 562 ảnh validation ở 18 ngưỡng từ 0.10 tới 0.95, bước 0.05. Ngưỡng 0.50: Dice pixel gộp 0.5345, IoU 0.3647, precision 0.4179, recall 0.7415; 53/282 ảnh bình thường có ít nhất một pixel dự đoán u. Ngưỡng 0.80 đạt Dice cao nhất trong lưới: 0.5707, IoU 0.3993, precision 0.5414, recall 0.6033; 48/282 ảnh bình thường có dự đoán u. Chọn ngưỡng dựa trên Dice validation khiến bỏ sót nhiều pixel hơn; không coi 0.80 là ngưỡng lâm sàng hoặc là hiệu quả đã xác nhận trên test. Job khảo sát: https://www.kaggle.com/code/vtan2405/xla-btxrd-segmenter-validation-audit.

Ảnh đã lưu ở 224×224; phóng lớn không khôi phục chi tiết ban đầu. Thử nghiệm tiếp theo nên giữ kiến trúc và so từng thay đổi trên validation, ví dụ cân bằng loss hoặc augmentation ảnh/mask đồng bộ; không thay đổi nhiều yếu tố cùng lúc hoặc tinh chỉnh bằng test.

## Thử trọng số foreground 50

Notebook `notebooks/kaggle_btxrd_segmenter_weight50/` chuẩn bị fine-tune 50 epoch từ checkpoint tốt nhất epoch 87, giảm trọng số foreground từ 100 xuống 50, giữ kiến trúc, loss, split, batch 4 và learning rate 2e-6. Checkpoint epoch 87 không có trạng thái AdamW nên optimizer và random được khởi tạo lại. Kết quả phải so với Dice 0,5345 của checkpoint gốc tại ngưỡng 0,5 trên cùng validation; ghi riêng Dice, IoU, precision, recall, ảnh thường dự đoán nhầm và tránh so trực tiếp loss khác trọng số. So sánh hai lần fine-tune không phải đối chứng hoàn chỉnh chỉ khác một biến: lịch sử optimizer và số epoch bổ sung khác. Không sử dụng test để chọn checkpoint hoặc ngưỡng.

## Pipeline theo source gốc

Pipeline `train_segmenter_official.py` dùng encoder, block và mask transformer của commit MIT ở trên; chỉ điều chỉnh import và chuyển checkpoint ViT AugReg qua API timm 1.0.31. Khởi tạo encoder `vit_base_patch16_384.augreg_in21k_ft_in1k` (nguồn: https://huggingface.co/timm/vit_base_patch16_384.augreg_in21k_ft_in1k, Apache-2.0), nội suy position embedding sang ảnh BTXRD 224×224; decoder hai lớp khởi tạo mới. Link checkpoint Segmenter ADE20K trong README upstream trả HTTP 404 khi kiểm tra, nên không khẳng định đã nạp weights segmentation của tác giả.

Giữ LayerNorm gốc, hai logit, pixel-wise cross-entropy không cân trọng số, SGD momentum 0.9, learning rate 0.001, weight decay 0 và polynomial decay power 0.9 đến 1e-5 như mặc định source. BTXRD cần adapter mask 0/1 và split riêng; ảnh nguồn đã là 224×224, train có lật ngang đồng bộ ảnh/mask. Pipeline không tự áp random crop và photometric distortion ADE20K lên X-quang; epoch và batch size cũng khác protocol ADE20K của paper. Đây là áp dụng kiến trúc và thuật toán train của source trên BTXRD, không phải tái lập benchmark ADE20K.
