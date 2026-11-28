# Thí nghiệm segmentation trên BTXRD

## So sánh ViT-B/16 có skip tích chập với EfficientViT-Seg-B1

Hai job dùng cùng membership đã kiểm tra theo checksum danh sách ảnh, gồm 1.302 ảnh có u để train, 562 ảnh validation và 561 ảnh test. Cả hai train 50 epoch, batch 8, seed 42, dùng chuẩn hóa ImageNet, lật ngang và xoay đồng bộ ảnh/mask, BCE trọng số foreground 10 cộng soft Dice. EfficientViT dùng backbone ImageNet-1K từ timm. ViT dùng torchvision ImageNet-1K, decoder kết hợp token 14×14 với feature tích chập 28×28, 56×56 và 112×112. ViT dùng AdamW lr 2e-5 cho backbone và 2e-4 cho decoder.

Chọn checkpoint bằng Dice pixel foreground gộp trên validation ở ngưỡng 0,5. Khi chấm test, trung bình xác suất ảnh gốc và ảnh lật ngang đã lật đầu ra về đúng chiều, sau đó áp ngưỡng 0,5. Job ViT hoàn tất 50 epoch. Checkpoint epoch 32 có Dice validation cao nhất là 0,666856.

| Mô hình | Dice pixel gộp | IoU pixel gộp | Dice trung bình trên ảnh có u | Ảnh thường dự đoán dưới 25 pixel foreground |
| --- | ---: | ---: | ---: | ---: |
| EfficientViT-Seg-B1 | 0,740377 | 0,587777 | 0,570889 | 69,04% |
| ViT-B/16 có skip tích chập | 0,705079 | 0,544495 | 0,527955 | 59,07% |

Báo cáo tải về tại `outputs/kaggle_checks/vit-detail/`. Kaggle Output có `comparison.json`, `vit_detail_test_metrics.json`, `history.json` và `best_vit_detail.pt`. [EfficientViT trên Kaggle](https://www.kaggle.com/code/qunthy/efficientvit-seg) · [ViT trên Kaggle](https://www.kaggle.com/code/vtan2405/xla-btxrd-vit-detail-vs-efficientvit) · [Notebook ViT](../notebooks/kaggle_btxrd_vit_detail/vit_detail.ipynb)

Web dùng checkpoint ViT epoch 32 với chuẩn hóa ImageNet, ngưỡng 0,5 và một lượt forward mỗi ảnh. Điểm Kaggle trong bảng dùng thêm lượt lật ngang.

```powershell
python scripts/build_vit_detail_notebook.py
kaggle kernels push -p notebooks/kaggle_btxrd_vit_detail --accelerator NvidiaTeslaT4
```

## ViT baseline cục bộ

Các lệnh dưới đây chạy baseline ViT-B/16 với đầu tích chập đơn giản và toàn bộ 2.613 ảnh train sau lọc. Job 20 epoch đầu tiên của cấu hình này ghi nhận Dice validation bằng 0. [Job ViT baseline](https://www.kaggle.com/code/vtan2405/xla-btxrd-vit-segmentation-20-epochs)

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m pip install -e ".[vit]"
.\.venv\Scripts\python.exe scripts/train_vit_segmentation.py train --archive data/raw/xla-seminar-uia/btxrd_final.zip --output outputs/vit_run1 --epochs 20 --batch-size 4
.\.venv\Scripts\python.exe scripts/train_vit_segmentation.py evaluate --archive data/raw/xla-seminar-uia/btxrd_final.zip --checkpoint outputs/vit_run1/best.pt --batch-size 4
```

Checkpoint tốt nhất được chọn trên validation. Dùng thư mục output mới cho từng lần train. Lệnh evaluate kiểm tra checksum của ZIP trước khi chấm test.

## Nguồn và cấu hình Segmenter

- Paper: Strudel và cộng sự, *Segmenter: Transformer for Semantic Segmentation*, ICCV 2021.
- Mã nguồn chính thức: https://github.com/rstrudel/segmenter, commit `20d1bfad354165ee45c3f65972a4d9c131f58d53`, giấy phép MIT (Robin Strudel và INRIA); bản license lưu cạnh mã trong `src/xla/vendor/segmenter/`.
- Decoder `MaskTransformer` và các block liên quan được lấy từ mã nguồn trên. Import được điều chỉnh cho package dự án và API `timm` hiện tại; phần decoder tuyến tính không được sử dụng. Block gốc ghi nhận dựa trên `timm` của Ross Wightman.
- Backbone là `timm` ViT-B/16 `vit_base_patch16_224.orig_in21k` pretrained ImageNet-21k, Apache-2.0 theo model card https://huggingface.co/timm/vit_base_patch16_224.orig_in21k. Đây không phải checkpoint Segmenter đã train trên ADE20K: decoder và hai class embedding được khởi tạo mới.
- Decoder dùng 2 lớp, 12 head, chiều embedding 768, MLP 3072, dropout 0.1, 2 class token nền/u, theo cấu hình Mask Transformer của source. Đầu ra hai lớp được biểu diễn bằng hiệu logit foreground và background để tái sử dụng pipeline nhị phân.
- Thí nghiệm dùng ảnh 224×224 BTXRD và 50 epoch, giữ split của baseline. Job đầu dùng loss BCE có trọng số foreground 10 cộng soft Dice và AdamW giống baseline trước, không phải loss cross-entropy và SGD mặc định của paper. Cấu hình này là thử nghiệm, không phải tái lập đầy đủ kết quả paper.

Job version 1 bị kẹt ở dự đoán gần như đồng nhất. Với hai lớp, `LayerNorm(2)` của decoder gốc có thể bão hòa khi chênh lệch logit lớn hơn epsilon. Job sửa dùng `mask_norm_eps=0.01` thay vì `1e-5`, learning rate `1e-5` và trọng số foreground `100`; kiến trúc và hai class token vẫn giữ nguyên. Trên hai ảnh train, thử nghiệm chỉ tối ưu decoder với feature backbone cố định cho Dice 0,196 sau 60 bước; đây không phải kết quả validation hoặc bằng chứng đảm bảo hội tụ. Cần đọc kết quả job mới để đánh giá thực tế. Epsilon được lưu trong checkpoint vì nó không nằm trong `state_dict`; checkpoint cũ mặc định dùng `1e-5`.

Notebook chạy Kaggle nằm trong `notebooks/kaggle_btxrd_segmenter/`; job độc lập với baseline cũ. `best.pt` được chọn theo Dice foreground trên validation, `final.pt` ở epoch cuối. Test không tham gia lựa chọn checkpoint.

### Đối chiếu source và kiểm tra job version 2

- So sánh AST xác nhận lớp `MaskTransformer` và `Block` được dùng trong dự án trùng với lớp tương ứng của commit upstream đã ghi ở trên; thay đổi import và epsilon được ghi riêng.
- [Source train gốc](https://github.com/rstrudel/segmenter/blob/20d1bfad354165ee45c3f65972a4d9c131f58d53/segm/engine.py) dùng cross-entropy theo pixel; [cấu hình train](https://github.com/rstrudel/segmenter/blob/20d1bfad354165ee45c3f65972a4d9c131f58d53/segm/train.py) mặc định dùng SGD và polynomial learning-rate scheduler. Các mặc định này áp dụng cho tập dữ liệu của tác giả, không bảo đảm tối ưu cho BTXRD.
- Version 2 dùng encoder `orig_in21k` của `timm`, decoder khởi tạo mới, BCE trọng số 100 cộng soft Dice, AdamW, epsilon `0.01`, ảnh 224×224 và không có augmentation. Đây không phải nguyên cấu hình upstream hay checkpoint pretrained segmentation của họ.
- Version 2 đạt Dice foreground gộp trên validation cao nhất 0,5014 ở epoch 46. Trên một mẫu thăm dò 16 ảnh validation (8 ảnh bình thường, 8 ảnh có u, chọn ngẫu nhiên seed 42), ngưỡng 0,5 cho TP 23.010, FP 18.945, FN 4.402 pixel; 1 trong 8 ảnh bình thường có dự đoán foreground. Mẫu nhỏ này chỉ gợi ý cần kiểm tra tô dư, không đại diện cho toàn tập.

### Train tiếp epoch 51–100

Notebook `notebooks/kaggle_btxrd_segmenter_continue/` đọc đúng `final.pt` epoch 50 của job version 2, đối chiếu SHA-256, lịch sử 50 epoch và checkpoint tốt nhất epoch 46. Kiến trúc, loss, positive weight 100, epsilon 0,01, batch 4 và split giữ nguyên; learning rate giảm từ 1e-5 xuống 2e-6 để thử giảm dao động. Đây là giả thuyết thử nghiệm, không bảo đảm cải thiện validation.

Checkpoint epoch 50 không có trạng thái optimizer hay random, nên AdamW và random được khởi tạo lại; đây là fine-tune tiếp từ trọng số, không phải khôi phục chính xác trạng thái train ban đầu. `best.pt` của job mới giữ nguyên checkpoint tốt nhất từ cả hai giai đoạn; `final.pt` là epoch 100, `history.json` bao gồm cả 100 epoch. `resume.pt` lưu optimizer và random state mỗi 5 epoch để có thể xây dựng quy trình phục hồi sau này.

### Kiểm tra ngưỡng trên validation

Checkpoint tốt nhất epoch 87 (SHA-256 `6c4846f0da78a4964a0fa08556bb0888d9e2683270398c1e718240c532a06a98`) được đo trên toàn bộ 562 ảnh validation ở 18 ngưỡng từ 0.10 tới 0.95, bước 0.05. Ngưỡng 0.50: Dice pixel gộp 0.5345, IoU 0.3647, precision 0.4179, recall 0.7415; 53/282 ảnh bình thường có ít nhất một pixel dự đoán u. Ngưỡng 0.80 đạt Dice cao nhất trong lưới: 0.5707, IoU 0.3993, precision 0.5414, recall 0.6033; 48/282 ảnh bình thường có dự đoán u. Chọn ngưỡng dựa trên Dice validation khiến bỏ sót nhiều pixel hơn; không coi 0.80 là ngưỡng lâm sàng hoặc là hiệu quả đã xác nhận trên test. Job khảo sát: https://www.kaggle.com/code/vtan2405/xla-btxrd-segmenter-validation-audit.

Ảnh đã lưu ở 224×224; phóng lớn không khôi phục chi tiết ban đầu. Thử nghiệm tiếp theo nên giữ kiến trúc và so từng thay đổi trên validation, ví dụ cân bằng loss hoặc augmentation ảnh/mask đồng bộ; không thay đổi nhiều yếu tố cùng lúc hoặc tinh chỉnh bằng test.

### Thử trọng số foreground 50

Notebook `notebooks/kaggle_btxrd_segmenter_weight50/` chuẩn bị fine-tune 50 epoch từ checkpoint tốt nhất epoch 87, giảm trọng số foreground từ 100 xuống 50, giữ kiến trúc, loss, split, batch 4 và learning rate 2e-6. Checkpoint epoch 87 không có trạng thái AdamW nên optimizer và random được khởi tạo lại. Kết quả phải so với Dice 0,5345 của checkpoint gốc tại ngưỡng 0,5 trên cùng validation; ghi riêng Dice, IoU, precision, recall, ảnh thường dự đoán nhầm và tránh so trực tiếp loss khác trọng số. So sánh hai lần fine-tune không phải đối chứng hoàn chỉnh chỉ khác một biến: lịch sử optimizer và số epoch bổ sung khác. Không sử dụng test để chọn checkpoint hoặc ngưỡng.

### Pipeline theo source gốc

Pipeline `train_segmenter_official.py` dùng encoder, block và mask transformer của commit MIT ở trên; chỉ điều chỉnh import và chuyển checkpoint ViT AugReg qua API timm 1.0.31. Khởi tạo encoder `vit_base_patch16_384.augreg_in21k_ft_in1k` (nguồn: https://huggingface.co/timm/vit_base_patch16_384.augreg_in21k_ft_in1k, Apache-2.0), nội suy position embedding sang ảnh BTXRD 224×224; decoder hai lớp khởi tạo mới. Link checkpoint Segmenter ADE20K trong README upstream trả HTTP 404 khi kiểm tra, nên không khẳng định đã nạp weights segmentation của tác giả.

Giữ LayerNorm gốc, hai logit, pixel-wise cross-entropy không cân trọng số, SGD momentum 0.9, learning rate 0.001, weight decay 0 và polynomial decay power 0.9 đến 1e-5 như mặc định source. BTXRD cần adapter mask 0/1 và split riêng; ảnh nguồn đã là 224×224, train có lật ngang đồng bộ ảnh/mask. Pipeline không tự áp random crop và photometric distortion ADE20K lên X-quang; epoch và batch size cũng khác protocol ADE20K của paper. Đây là áp dụng kiến trúc và thuật toán train của source trên BTXRD, không phải tái lập benchmark ADE20K.

### Audit checkpoint hiện tại trên hai ảnh train u nhỏ

Script `scripts/audit_small_lesions.py` kiểm tra checkpoint weight 10 epoch 15 SHA-256 `ea9eb43856adb5fd20a4d430aa2afb2dedf6156fa9f76e1d458ff6297ec1a056`, dữ liệu SHA-256 cố định và code gốc commit `20d1bfad354165ee45c3f65972a4d9c131f58d53`. Hai mẫu là hai mask train đầu tiên có 16–255 pixel foreground, không chọn theo điểm hoặc dùng test. Máy audit dùng CPU. Báo cáo và ảnh đối chiếu được lưu tại `outputs/small-lesion-audit/` và `outputs/small-lesion-epsilon-audit/`.

- Tải nguyên văn decoder và block từ commit upstream; điều chỉnh import để chạy bằng timm hiện tại và dùng cùng hàm khởi tạo. Với cùng checkpoint, epsilon mask 0,01 và đặc trưng trên hai ảnh, decoder gốc và decoder dự án chênh lệch tối đa **0,0**; xác suất sigmoid hiệu hai logit so với softmax hai lớp chênh lệch **5,96e-8**. Đây là kiểm tra forward trên hai ảnh, không chứng minh mọi trường hợp trùng nhau.
- Encoder timm hiện tại và encoder Segmenter vendor, nạp cùng trọng số, chênh đặc trưng tối đa **0,002999**. Epsilon LayerNorm của timm là 1e-6, encoder vendor là 1e-5; khi khớp epsilon cả 25 lớp, sai khác giảm còn **1,66e-5**. Với epsilon mặc định từng encoder, xác suất sau decoder chênh tối đa **0,0003094**, và **0 pixel** đổi phân loại tại ngưỡng 0,7 trên hai ảnh này. Không khẳng định các ảnh hoặc ngưỡng khác sẽ không bị ảnh hưởng; không tự thay epsilon của checkpoint đã train.
- Decoder có gradient hữu hạn, khác 0 trên các tham số được theo dõi. Thử tối ưu riêng decoder, giữ backbone đông cứng, decoder ở eval, không augmentation, weighted BCE + batch Dice, AdamW lr 2e-5, 60 bước, ngưỡng 0,7. Ảnh `IMG000370.jpeg` (31 pixel u) Dice **0 → 0**, sau đó dự đoán rỗng. Ảnh `IMG000438.jpeg` (180 pixel u) Dice **0,5724 → 0,6212**. Loss giảm 1,01318 → 1,01065. Đây là thử nghiệm hai ảnh, cấu hình hạn chế, không chứng minh model không thể overfit khi train cả backbone hoặc với loss/siêu tham số khác.

Đối chiếu mới xác nhận adapter/decoder trên mẫu thử, chưa tìm ra nguyên nhân chính của Dice thấp. Bước kế tiếp là thử overfit toàn model trên vài ảnh với ngưỡng 0,5 và 0,7, theo dõi gradient encoder/decoder và đầu ra từng ảnh; đánh giá nhiều lựa chọn và hạt giống trên validation trước khi chốt test.

Đã thử tiếp toàn bộ encoder + decoder trên chính hai ảnh đó, eval mode để bỏ dropout, cùng loss, AdamW lr 2e-6, 20 bước. Ở ngưỡng 0,7, Dice ảnh 31 pixel vẫn 0; ảnh 180 pixel tăng 0,5724 → 0,5828. Gradient patch embedding của encoder và tham số decoder đều hữu hạn, khác 0; loss 1,01318 → 1,01146. Kết quả lưu `outputs/small-lesion-full-model/`. Thử nghiệm chưa đo ngưỡng 0,5 và chưa đủ bước/ảnh để kết luận khả năng overfit; cần tiếp tục kiểm tra nhiều ngưỡng và cấu hình trên GPU.

## Thử cải tiến ViT bằng fine-tune có đối chứng

Nguồn khởi tạo: checkpoint tốt nhất của job weight 50, SHA-256 `97ba621bcdd8c6c5386a1a2e0c47fd618b7232f2c1d929bf350e18ed230886e8`. Dice validation đã ghi nhận là 0,5677, IoU 0,3964, ngưỡng 0,5. Đây là Segmenter với backbone ViT-B/16, không phải đầu convolution của baseline 20 epoch ban đầu.

### Giả thuyết và cấu hình

Các lần khảo sát trước cho thấy tăng ngưỡng giúp giảm tô dư. Trọng số foreground cao trong BCE là một nguyên nhân cần kiểm tra: [tài liệu PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) mô tả trọng số dương ảnh hưởng đến cân bằng precision/recall. Giảm trọng số có thể giảm false positive nhưng cũng có thể tăng bỏ sót; chưa có bằng chứng rằng thí nghiệm mới tốt hơn.

Hai nhánh bắt đầu từ cùng checkpoint và chạy 20 epoch mỗi nhánh:

- Đối chứng: giữ positive weight 50.
- Ứng viên: giảm positive weight xuống 10.

Cả hai dùng cùng seed 42, batch 4, AdamW khởi tạo lại, learning rate 2e-6, weight decay 0,01 và loss BCE + soft Dice. Ảnh/mask train có lật ngang, xoay tối đa 7 độ, dịch tối đa 4 pixel và scale 0,97–1,03 đồng bộ; ảnh nội suy bilinear, mask dùng nearest và giữ giá trị 0/1. Validation không augmentation. Hai nhánh chỉ khác positive weight; so với checkpoint nguồn còn khác augmentation và số epoch, nên không quy toàn bộ thay đổi về một nguyên nhân.

### Chọn kết quả

Chọn checkpoint mỗi nhánh bằng Dice foreground gộp trên validation ở ngưỡng 0,5. Giữ checkpoint nguồn nếu không vượt được điểm nguồn. Sau đó khảo sát các checkpoint đã chọn trên cùng lưới ngưỡng 0,3–0,9, bước 0,1, ghi Dice, IoU, precision, recall và số ảnh bình thường có dự đoán u. `comparison.json` chứa cả nguồn và hai nhánh; `validation_winner` có thể vẫn là `source`.

Không đánh giá test trong job này. Kết quả calibration chỉ là kết quả validation; cần chốt checkpoint và ngưỡng trước khi đánh giá test. Demo đang chạy không tự đổi checkpoint khi job kết thúc.

Mỗi nhánh lưu `best.pt`, `final.pt`, `resume.pt`, `config.json`, `history.json`. Resume snapshot có optimizer và RNG nhưng script này chưa cung cấp lệnh khôi phục tự động. File cũ và checkpoint nguồn được giữ nguyên.

Code: `scripts/improve_segmenter.py`, `src/xla/data/augmentation.py`. Notebook tiếng Anh: `notebooks/kaggle_btxrd_vit_improvement/vit_improvement_ab.ipynb`. [Job Kaggle](https://www.kaggle.com/code/vtan2405/xla-btxrd-vit-improvement-ab-20-epochs).

### Kết quả đã kiểm tra ngày 27/11/2026

Kaggle báo COMPLETE, cả hai nhánh đủ 20 epoch. Đối chiếu `comparison.json` và `history.json` trên 562 ảnh validation (282 bình thường, 280 có u):

| Checkpoint | Ngưỡng | Dice | IoU |
| --- | --- | --- | --- |
| Nguồn weight 50 | 0,5 | 0,567719 | 0,396374 |
| Weight 10, epoch 15 | 0,5 | 0,607104 | 0,435857 |
| Nguồn weight 50 | 0,7 | 0,581604 | 0,410044 |
| Weight 10, epoch 15 | 0,7 | 0,623782 | 0,453259 |

Ở cùng ngưỡng 0,7, Dice tăng 4,22 điểm phần trăm; số ảnh bình thường có dự đoán foreground giảm từ 43 xuống 33. Nhánh weight 50 không vượt checkpoint nguồn, nên `best.pt` của nhánh đó vẫn là checkpoint nguồn. Nhánh weight 10 chọn epoch 15, không chọn epoch cuối: Dice validation epoch 20 là 0,595548 tại ngưỡng 0,5. Dice train epoch 20 là 0,708385.

`validation_winner` là `weight_10`. Demo cục bộ từng dùng `weight_10/best.pt` với ngưỡng 0,7. Báo cáo gốc được giữ tại `outputs/kaggle_checks/vit-improvement-ab/vit_improvement_ab/`. Chưa có điểm trên toàn tập test cho checkpoint này.

Checkpoint tải về có kích thước 407.054.362 byte, SHA-256 `ea9eb43856adb5fd20a4d430aa2afb2dedf6156fa9f76e1d458ff6297ec1a056`. Đã đối chiếu metadata epoch 15, positive weight 10, checksum dataset và checksum checkpoint nguồn với báo cáo Kaggle trước khi nạp vào backend.

### Thử nghiệm tiếp: Dice theo batch và theo ảnh

Job riêng: [XLA BTXRD ViT Image Dice AB 20 Epochs](https://www.kaggle.com/code/vtan2405/xla-btxrd-vit-image-dice-ab-20-epochs). Notebook nằm tại `notebooks/kaggle_btxrd_vit_loss_ab/vit_loss_ab.ipynb`; đầu vào là output job A/B weight ở trên, chọn đúng `weight_10/best.pt` bằng SHA-256.

Giữ ViT-B/16, Mask Transformer, epsilon 0,01, ảnh 224×224, positive weight 10, BCE trung bình pixel, seed 42, batch 4, learning rate 2e-6, AdamW weight decay 0,01 và augmentation cặp ảnh/mask. Chỉ thay cách gộp phần Dice:

- `dice_batch`: cộng giao và diện tích trên toàn batch như loss hiện tại.
- `dice_image`: tính soft Dice từng ảnh rồi lấy trung bình; vẫn tính cả ảnh bình thường và giữ BCE để phạt false positive. Smoothing bằng 1 như nhánh đối chứng.

Mỗi nhánh chạy 20 epoch từ cùng checkpoint epoch 15, khởi tạo lại optimizer và RNG. Đây là thay đổi loss do dự án thử nghiệm, không thay decoder và không tuyên bố tái lập protocol train của paper. Giả thuyết là giảm ảnh hưởng áp đảo của vùng u lớn lên phần Dice; chưa có kết quả xác nhận hiệu quả.

Chọn `best.pt` theo Dice foreground pixel gộp trên validation ở ngưỡng 0,5, giữ nguyên nguồn nếu không vượt; `final.pt` lưu epoch 20, `resume.pt` có optimizer/RNG mỗi 5 epoch. Khảo sát nguồn và checkpoint được chọn ở cùng ngưỡng 0,3–0,9. Báo cáo thêm nhóm tổn thương dưới 256 pixel và từ 256 pixel trở lên: số ảnh, TP/FP/FN, Dice pixel gộp và Dice trung bình theo ảnh; ảnh bình thường tính riêng số ảnh có dự đoán u. Ngưỡng và lựa chọn checkpoint chỉ dựa trên validation, không chạy đánh giá test.

Notebook đóng gói 17 file nguồn kèm license upstream MIT; dependency giữ `timm==1.0.31`, `einops==0.8.1` như job trước. Không thêm mã thư viện bên thứ ba mới. GPU Kaggle phải khả dụng. Lệnh gửi job:

```powershell
python -m kaggle kernels push -p notebooks/kaggle_btxrd_vit_loss_ab --accelerator NvidiaTeslaT4
```

### Kết quả Dice loss A/B kiểm tra ngày 28/11/2026

Kaggle API báo `COMPLETE`. Log và hai file `history.json` xác nhận mỗi nhánh hoàn tất đủ epoch 1–20. Báo cáo tải về tại `outputs/kaggle_checks/vit-loss-ab/vit_loss_ab/`; log tại `outputs/kaggle_checks/vit-loss-ab/xla-btxrd-vit-image-dice-ab-20-epochs.log`. Đã kiểm tra lại công thức Dice/IoU từ TP/FP/FN ở cả 7 ngưỡng, số ảnh validation, kết quả nguồn khớp từng bộ đếm với job weight 10 trước, và điểm checkpoint chọn tại ngưỡng 0,5 khớp epoch tốt nhất trong lịch sử. Không tải checkpoint mới về vì báo cáo vẫn chọn nguồn.

Các chỉ số dưới đây là **foreground pixel gộp trên 562 ảnh validation**, không phải Dice trung bình theo ảnh. Ngưỡng tốt nhất lấy trong lưới 0,3–0,9:

| Checkpoint | Epoch tốt nhất tại ngưỡng 0,5 | Dice tại 0,5 | Ngưỡng tốt nhất | Dice tại ngưỡng tốt nhất | IoU tại ngưỡng tốt nhất | Ảnh normal có dự đoán / 282 |
| --- | --- | --- | --- | --- | --- | --- |
| Nguồn weight 10 đang dùng | 15 của job trước | 0,607104 | 0,7 | 0,623782 | 0,453259 | 33 |
| Fine-tune `dice_batch` | 15 | 0,617611 | 0,6 | 0,623337 | 0,452788 | 32 |
| Fine-tune `dice_image` | 18 | 0,612869 | 0,6 | 0,621430 | 0,450778 | 54 |

Tại ngưỡng 0,5, hai nhánh mới đều vượt nguồn. Sau calibration, `validation_winner` vẫn là `source`; nhánh batch thấp hơn nguồn khoảng 0,000446 Dice (0,0446 điểm phần trăm), nhánh image thấp hơn khoảng 0,002353 (0,2353 điểm phần trăm). Đây là chênh lệch quan sát từ một seed, không phải kết luận về ý nghĩa thống kê.

Để đối chiếu riêng hai cách tính loss ở **cùng ngưỡng 0,6**, trên 104 ảnh có tổn thương dưới 256 pixel, Dice trung bình theo ảnh tăng từ 0,152926 (`dice_batch`) lên 0,174607 (`dice_image`); Dice pixel gộp nhóm này tăng từ 0,197262 lên 0,199171. Tuy vậy, số ảnh normal có dự đoán tăng từ 32 lên 54; Dice pixel toàn validation giảm từ 0,623337 xuống 0,621430. Kết quả có cải thiện ở nhóm u nhỏ nhưng chưa cải thiện tiêu chí chọn model toàn validation.

Epoch cuối không phải checkpoint tốt nhất: Dice validation epoch 20 tại ngưỡng 0,5 là 0,604257 (batch) và 0,599607 (image). Tiếp tục giữ demo ở checkpoint nguồn SHA-256 `ea9eb43856adb5fd20a4d430aa2afb2dedf6156fa9f76e1d458ff6297ec1a056`, ngưỡng 0,7. Job này không đánh giá test và không tự thay checkpoint demo.

### Thí nghiệm crop u nhỏ A/B ngày 28/11/2026

Giả thuyết: thay đổi cách nhìn tổn thương nhỏ trong lúc train có thể cải thiện nhóm mà model nguồn còn yếu. Tại ngưỡng 0,7, model nguồn có Dice pixel gộp 0,178223 trên 104 ảnh validation với vùng u dưới 256 pixel, so với 0,660242 trên 176 ảnh có u còn lại. Chưa có kết quả xác nhận crop giúp cải thiện.

Hai nhánh dùng cùng checkpoint nguồn weight 10 nêu trên, ViT-B/16 + Mask Transformer epsilon 0,01, weighted BCE foreground 10 cộng soft Dice theo batch, seed 42, batch 4, AdamW khởi tạo lại, learning rate cố định 2e-6, weight decay 0,01, paired augmentation và 20 epoch:

- `full_image`: chỉ paired augmentation hiện tại, không crop.
- `lesion_crop`: sau paired augmentation, với mask có 1–255 pixel foreground, xác suất 0,5 thực hiện crop vuông rồi resize về 224×224; các lần còn lại giữ nguyên ảnh. Ảnh normal và mask từ 256 pixel trở lên luôn giữ toàn ảnh. Không thêm, loại hoặc nhân bản mẫu.

Crop bao trọn mask sau augmentation và vùng ngữ cảnh 8 pixel mỗi phía, giới hạn theo biên ảnh. Cạnh crop ngẫu nhiên 112–196 pixel, được nới nếu cần để chứa trọn vùng trên; nếu không có crop nhỏ hơn ảnh phù hợp thì giữ nguyên. Ảnh dùng bilinear với antialias, mask dùng nearest. Có nghĩa là giữ đủ foreground **sau augmentation**, không bảo đảm phép affine trước đó chưa cắt foreground ở mép ảnh. Ảnh gốc chỉ có 224×224 nên zoom không khôi phục chi tiết chưa có; thí nghiệm chỉ thay tỷ lệ tổn thương mà model nhìn thấy. Không dùng ground truth để crop khi suy luận.

Validation luôn chạy trên cùng 562 ảnh nguyên, không augmentation/crop; test không được đánh giá. Chọn `best.pt` bằng Dice foreground pixel gộp trên validation tại ngưỡng 0,5, giữ nguồn nếu không vượt. Sau train, khảo sát nguồn và best của hai nhánh tại 0,3–0,9, báo cáo Dice/IoU, precision/recall, số ảnh normal có dự đoán và hai nhóm kích thước u. Cần so cả cùng ngưỡng lẫn ngưỡng tốt nhất, đồng thời kiểm tra nhóm u nhỏ và false positive. Dice train được đo trên hai loại ảnh augmentation khác nhau nên không dùng để quyết định nhánh thắng.

Mỗi nhánh lưu `best.pt`, `final.pt`, `resume.pt`, `config.json`, `history.json`. Cấu hình crop và việc validation dùng ảnh nguyên được lưu trong metadata; `comparison.json` đánh dấu `experiment=small_lesion_crop_ab`. `resume.pt` có optimizer/RNG nhưng chưa có lệnh resume tự động. Job không tự thay checkpoint của demo.

Code: `src/xla/data/lesion_crop.py`, `scripts/improve_segmenter.py`. Notebook tự chứa 18 file nguồn và license upstream MIT: `notebooks/kaggle_btxrd_vit_lesion_crop_ab/vit_lesion_crop_ab.ipynb`. Đóng gói lại bằng `python scripts/build_lesion_crop_notebook.py`; kiểm tra notebook khớp source bằng `tests/test_lesion_crop_notebook.py`. Dependency giữ `timm==1.0.31`, `einops==0.8.1`, không thêm thư viện bên thứ ba.

Job riêng: [XLA BTXRD ViT Lesion Crop AB 20 Epochs](https://www.kaggle.com/code/vtan2405/xla-btxrd-vit-lesion-crop-ab-20-epochs), gắn output notebook weight A/B để tìm đúng checkpoint nguồn theo SHA-256. Lệnh gửi:

```powershell
python -m kaggle kernels push -p notebooks/kaggle_btxrd_vit_lesion_crop_ab --accelerator NvidiaTeslaT4
```

Đã gửi version 1 ngày 28/11/2026 bằng lệnh trên; Kaggle API xác nhận `RUNNING` tại lần kiểm tra sau gửi. Chưa có kết quả cải thiện. Trước gửi: 33 test liên quan pass, Ruff pass, bundle notebook khớp source và import độc lập thành công. Kiểm tra trên dataset thật giữ split 2.613/562/561; ba mẫu u nhỏ sau crop tăng diện tích mask 33→77, 179→363 và 41→82 pixel, mẫu normal vẫn rỗng. Ảnh kiểm tra lưu cục bộ ở `outputs/crop-ab-qa/preview.png`.

#### Kết quả crop A/B

Kaggle báo `COMPLETE`, mỗi nhánh có đủ 20 epoch. Đã tải log, config, lịch sử và `comparison.json` về `outputs/kaggle_checks/vit-lesion-crop-ab/`; đối chiếu checksum nguồn, số ảnh, TP/FP/FN, Dice/IoU và lịch sử checkpoint. Ở cùng ngưỡng 0,7 (cũng là ngưỡng tốt nhất trong lưới của cả ba):

| Model | Epoch chọn | Dice pixel validation | IoU | Dice pixel u nhỏ | Ảnh normal dự đoán u / 282 |
| --- | --- | --- | --- | --- | --- |
| Nguồn đang dùng | 15 job trước | 0,623782 | 0,453259 | 0,178223 | 33 |
| Fine-tune ảnh nguyên | 18 | 0,627730 | 0,457439 | 0,196123 | 34 |
| Fine-tune crop | 12 | 0,620514 | 0,449816 | 0,184866 | 26 |

`validation_winner` là `full_image`, cao hơn nguồn 0,003948 Dice (0,395 điểm phần trăm). Nhánh crop thấp hơn nguồn 0,003268 Dice tổng nhưng giảm số ảnh normal bị tô dư từ 33 xuống 26. So với nhánh full-image cùng ngưỡng, crop thấp hơn về Dice tổng và nhóm u nhỏ, nhưng giảm ảnh normal bị tô dư từ 34 xuống 26. Đây là một thử nghiệm với một seed; chênh lệch nhỏ chưa chứng minh khả năng tổng quát hóa. Điểm train 0,728326 ở epoch 18 không phải điểm validation. Chưa đánh giá test; checkpoint mới chưa được tải hoặc thay vào demo.

### Thử thêm loss Dice cho u nhỏ

Sau audit hai ảnh train, chạy job riêng [ViT Small Dice A/B](https://www.kaggle.com/code/vtan2405/xla-btxrd-vit-small-dice-ab-20-epochs). Hai nhánh cùng checkpoint `ea9eb438...`, cùng dữ liệu, augmentation, seed 42, AdamW lr 2e-6, batch 4 và 20 epoch. Đối chứng dùng weighted BCE + batch Dice hiện tại. Nhánh `small_dice` cộng 0,5 lần Dice loss trung bình theo ảnh cho mask sau augmentation có 1–255 pixel; batch không có ảnh đủ điều kiện thì không cộng. Normal và mask lớn vẫn tham gia loss gốc; validation/test không được crop. Chỉ là giả thuyết, không bảo đảm cải thiện và có thể tăng tô nhầm ảnh normal.

Chọn checkpoint bằng Dice pixel foreground trên validation ở ngưỡng 0,5 như job trước, giữ nguồn nếu không vượt. Audit cùng các ngưỡng 0,3–0,9 và báo cáo riêng nhóm u nhỏ, ảnh normal tô dư. Loss validation hai nhánh khác định nghĩa nên không so trực tiếp. Job không chạy test và không tự đổi model demo. Notebook `notebooks/kaggle_btxrd_vit_small_dice_ab/vit_small_dice_ab.ipynb` được tạo bằng `python scripts/build_small_dice_notebook.py`. Đã gửi version 1 lên Kaggle T4; API báo `RUNNING` ngay sau khi gửi. Trước gửi 17 test pass; 18 file nhúng khớp source và các cell compile. Chưa có kết quả.

#### Kết quả Small Dice A/B đã kiểm tra

Kaggle báo `COMPLETE`, cả hai nhánh đủ epoch 1–20. Đã tải log, config, lịch sử và `comparison.json` về `outputs/kaggle_checks/vit-small-dice-ab/`. Checksum nguồn và toàn bộ số liệu nguồn khớp job trước; kiểm tra lại Dice/IoU từ TP/FP/FN, hai nhóm kích thước và số ảnh validation. Hai nhánh cùng chọn epoch 15 bằng Dice validation tại ngưỡng 0,5: control 0,617003, small_dice 0,608942. Trọng số phần loss bổ sung được xác nhận lần lượt 0 và 0,5.

Ở cùng ngưỡng 0,7 trên 562 ảnh validation:

| Checkpoint | Dice tổng | IoU | Dice pixel nhóm u nhỏ | Dice trung bình ảnh nhóm u nhỏ | Normal tô dư / 282 |
| --- | --- | --- | --- | --- | --- |
| Nguồn đang dùng | 0,623782 | 0,453259 | 0,178223 | 0,142577 | 33 |
| Control của job này | 0,620871 | 0,450191 | 0,201788 | 0,153886 | 32 |
| Small Dice | 0,615963 | 0,445048 | 0,204189 | 0,163859 | 38 |

Sau khảo sát ngưỡng, control tốt nhất là 0,623077 tại 0,6; Small Dice tốt nhất là 0,616206 tại 0,6. `validation_winner` vẫn là `source`. Nhánh ảnh nguyên của job crop trước đó vẫn là ứng viên cao nhất đã ghi nhận, Dice 0,627730 tại 0,7; chưa nạp vào demo và chưa đánh giá test. Không coi chênh lệch giữa các lần chạy là có ý nghĩa thống kê vì chưa lặp nhiều seed.

Small Dice cải thiện nhẹ nhóm u nhỏ tại cùng ngưỡng 0,7, nhưng không vượt control về Dice tổng và tăng ảnh normal bị tô dư từ 32 lên 38. Lần thử này chưa ủng hộ thay model bằng nhánh Small Dice. Epoch cuối Small Dice có Dice validation 0,596537 ở ngưỡng 0,5, không phải checkpoint được chọn. Không tải hoặc thay checkpoint demo trong lần kiểm tra kết quả này.
