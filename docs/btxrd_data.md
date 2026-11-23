# Kiểm tra và chia dữ liệu BTXRD

Nguồn: `data/raw/xla-seminar-uia/btxrd_final.zip`, revision `cf8b0569b18e43e7cecab412a06e60aeaf258b32` của [dataset Hugging Face](https://huggingface.co/datasets/torusama/xla-seminar-uia). SHA-256: `59ce94244fdf997f0c68e6a378690a4359e6c411f7fda7ac2f3994489637884b`.

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
python scripts/prepare_btxrd_splits.py --output data/splits/btxrd
```

Lệnh train ViT dùng cùng hàm `make_splits` tạo manifest, không đọc CSV trực tiếp. Khi triển khai EfficientViT, dùng cùng danh sách hoặc cùng hàm này. Chỉ dùng validation để chọn model; giữ test cho bước đánh giá cuối.
