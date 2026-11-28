# Kiểm tra và chia dữ liệu BTXRD

BoneScope tải `btxrd_final.zip` từ Kaggle và lưu ở `data/raw/xla-seminar-uia/`. Gói Kaggle giữ nguyên archive đóng gói từ Hugging Face revision `cf8b0569b18e43e7cecab412a06e60aeaf258b32`. SHA-256 của ZIP là `59ce94244fdf997f0c68e6a378690a4359e6c411f7fda7ac2f3994489637884b`. [Gói dữ liệu trên Kaggle](https://www.kaggle.com/datasets/vtan2405/bonescope-btxrd-splits) · [Dữ liệu gốc CC BY 4.0](https://api.figshare.com/v2/articles/27865398)

## Kiểm tra và chuẩn hóa

Đã đọc toàn bộ 3.746 ảnh: đều là JPEG RGB 224 × 224, đọc được và có annotation tương ứng. Kích thước annotation khớp ảnh, nhãn khớp thư mục, polygon có ít nhất ba điểm trong khung ảnh. 1.879 ảnh `Normal` không có polygon; 1.867 ảnh u có polygon. Đây là kiểm tra cấu trúc, không xác minh độ chính xác lâm sàng hay chất lượng đường bao bằng mắt.

Ảnh trong ZIP là JPEG 8-bit; chưa có bằng chứng xác nhận phép xử lý cường độ nào đã được áp dụng trước khi đóng gói. Loader chuyển pixel sang float `[0, 1]`, sau đó áp dụng `(x - 0.5) / 0.5` cho ba kênh. Đây là chuẩn hóa theo cấu hình pretrained ViT, không phải theo thống kê riêng của dataset. Trên tập train sau lọc, mean RGB lần lượt là 0.268945, 0.268940, 0.268951 và độ lệch chuẩn là 0.240746, 0.240741, 0.240751, tính trên thang `[0, 1]`. Mask được tạo từ polygon, có giá trị 0 hoặc 1.

## Split dùng chung

Giữ split trong ZIP. Hai nhóm ảnh trùng byte có nhãn khác nhau, mỗi nhóm gồm một ảnh `Normal` và một ảnh `osteochondroma`: loại cả bốn ảnh. Sau đó loại thêm sáu ảnh train trùng byte với validation/test. File ZIP không bị thay đổi.

| Tập | Bình thường | Có u | Tổng |
|---|---:|---:|---:|
| Train | 1.311 | 1.302 | 2.613 |
| Validation | 282 | 280 | 562 |
| Test | 281 | 280 | 561 |

Vẫn có ảnh trùng byte trong cùng một split. Các split không chia sẻ hash ảnh; kiểm tra này không phát hiện bản sao đã nén lại hoặc ảnh khác của cùng bệnh nhân. Không có đủ thông tin để xác nhận chia theo bệnh nhân.

`data/splits/btxrd/{train,val,test}.csv` chứa đường dẫn ảnh/annotation bên trong ZIP, SHA-256, nhãn gốc và nhãn nhị phân. `excluded.csv` liệt kê ảnh bị loại cùng lý do; `audit.json` ghi hash nguồn, số lượng và thống kê ảnh. Những file này được Git bỏ qua; để tái tạo, dùng output chưa tồn tại:

```powershell
$env:PYTHONUTF8 = "1"
python scripts/prepare_btxrd_splits.py --output data/splits/btxrd-regenerated
```

Các notebook tạo split từ archive bằng `make_splits`. Job ViT so trực tiếp với EfficientViT còn kiểm tra SHA-256 danh sách ảnh sau khi giới hạn train vào 1.302 mẫu có u, đúng như job EfficientViT gốc. Chọn checkpoint trên validation và chấm test sau khi chốt checkpoint.

## Tập so sánh trên web

BoneScope kiểm tra định dạng RGB 224×224, SHA-256 từng ảnh, kích thước annotation, tọa độ polygon, nhãn, mask và ảnh đồng nhất màu trước khi chạy model. Khi nhiều dòng có cùng SHA-256, giữ dòng hợp lệ đầu tiên theo manifest. Quy trình loại một bản trùng byte, còn 560 ảnh gồm 281 ảnh bình thường và 279 ảnh có u. Danh sách dùng chung cho hai model và được lưu trong `selection.json` của báo cáo ZIP.
