# Thử cải tiến ViT bằng fine-tune có đối chứng

Nguồn khởi tạo: checkpoint tốt nhất của job weight 50, SHA-256 `97ba621bcdd8c6c5386a1a2e0c47fd618b7232f2c1d929bf350e18ed230886e8`. Dice validation đã ghi nhận là 0,5677, IoU 0,3964, ngưỡng 0,5. Đây là Segmenter với backbone ViT-B/16, không phải đầu convolution của baseline 20 epoch ban đầu.

## Giả thuyết và cấu hình

Các lần khảo sát trước cho thấy tăng ngưỡng giúp giảm tô dư. Trọng số foreground cao trong BCE là một nguyên nhân cần kiểm tra: [tài liệu PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) mô tả trọng số dương ảnh hưởng đến cân bằng precision/recall. Giảm trọng số có thể giảm false positive nhưng cũng có thể tăng bỏ sót; chưa có bằng chứng rằng thí nghiệm mới tốt hơn.

Hai nhánh bắt đầu từ cùng checkpoint và chạy 20 epoch mỗi nhánh:

- Đối chứng: giữ positive weight 50.
- Ứng viên: giảm positive weight xuống 10.

Cả hai dùng cùng seed 42, batch 4, AdamW khởi tạo lại, learning rate 2e-6, weight decay 0,01 và loss BCE + soft Dice. Ảnh/mask train có lật ngang, xoay tối đa 7 độ, dịch tối đa 4 pixel và scale 0,97–1,03 đồng bộ; ảnh nội suy bilinear, mask dùng nearest và giữ giá trị 0/1. Validation không augmentation. Hai nhánh chỉ khác positive weight; so với checkpoint nguồn còn khác augmentation và số epoch, nên không quy toàn bộ thay đổi về một nguyên nhân.

## Chọn kết quả

Chọn checkpoint mỗi nhánh bằng Dice foreground gộp trên validation ở ngưỡng 0,5. Giữ checkpoint nguồn nếu không vượt được điểm nguồn. Sau đó khảo sát các checkpoint đã chọn trên cùng lưới ngưỡng 0,3–0,9, bước 0,1, ghi Dice, IoU, precision, recall và số ảnh bình thường có dự đoán u. `comparison.json` chứa cả nguồn và hai nhánh; `validation_winner` có thể vẫn là `source`.

Không đánh giá test trong job này. Kết quả calibration chỉ là kết quả validation; cần chốt checkpoint và ngưỡng trước khi đánh giá test. Demo đang chạy không tự đổi checkpoint khi job kết thúc.

Mỗi nhánh lưu `best.pt`, `final.pt`, `resume.pt`, `config.json`, `history.json`. Resume snapshot có optimizer và RNG nhưng script này chưa cung cấp lệnh khôi phục tự động. File cũ và checkpoint nguồn được giữ nguyên.

Code: `scripts/improve_segmenter.py`, `src/xla/data/augmentation.py`. Notebook tiếng Anh: `notebooks/kaggle_btxrd_vit_improvement/vit_improvement_ab.ipynb`. [Job Kaggle](https://www.kaggle.com/code/vtan2405/xla-btxrd-vit-improvement-ab-20-epochs).
