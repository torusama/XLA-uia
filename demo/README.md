# Demo ViT và EfficientViT

## Cấu trúc

```text
demo/
  frontend/              Giao diện HTML, CSS và JavaScript
  backend/dataset.py     Đọc ảnh test và tạo mask ground truth từ polygon
  backend/models.py      Nạp checkpoint ViT, chạy model và đo chỉ số
  backend/server.py      HTTP API và server cục bộ
  backend/export_samples.py  Xuất ảnh/mask để xem trên bản web tĩnh
  build_site.py          Tạo thư mục dist từ frontend và ảnh test
```

## Chạy cục bộ

Chuẩn bị môi trường theo README gốc, tải dataset và tạo split BTXRD. Trong thư mục gốc repo:

```powershell
python -m demo.backend.server
```

Mở `http://127.0.0.1:8765`. Nếu cổng đang được dùng bởi server cũ, dừng server đó hoặc chạy `python -m demo.backend.server --port 8766` rồi mở `http://127.0.0.1:8766`.

Giao diện tự lấy 561 ảnh trong `data/splits/btxrd/test.csv`. Nút trước/sau đổi ảnh cho tất cả các tab. Ảnh và ground truth được đọc từ `data/raw/xla-seminar-uia/btxrd_final.zip`; server kiểm tra SHA-256 của ZIP. Mỗi khi đổi ảnh hoặc tab, model đã sẵn sàng sẽ tự chạy lại; tab So sánh chạy hai model trên cùng một ảnh.

## Nối model

Khi đã tải checkpoint ViT thực tế từ Kaggle, chỉ đường dẫn tới `best.pt` hoặc `final.pt`:

```powershell
python -m demo.backend.server --vit-checkpoint outputs/kaggle_vit_20e/btxrd_vit_20e/best.pt
```

Checkpoint trong ví dụ được tải từ job Kaggle ViT 20 epoch. Dice validation của job này bằng 0 trong cả 20 epoch, nên chỉ dùng để kiểm tra đường đi dữ liệu và giao diện; không coi mask là kết quả có chất lượng. Server dùng `torch.load(..., weights_only=True)`, kiểm tra checkpoint thuộc đúng ZIP BTXRD và chọn kiến trúc phù hợp với metadata checkpoint. Ảnh được chuẩn hóa mean/std 0.5 như khi train; mask foreground dùng ngưỡng 0.5. Server chọn CUDA khi có, nếu không dùng CPU. Không thay thế checkpoint bằng trọng số ngẫu nhiên.

EfficientViT chưa được triển khai trong repo. Khi đã có kiến trúc và checkpoint, bổ sung cách nạp trong `backend/models.py` và đăng ký model dưới tên `efficientvit`; model cần nhận tensor `[B, 3, 224, 224]` và trả logits `[B, 1, 224, 224]`. Hai model phải cùng cách tiền xử lý, hậu xử lý và split. Không đặt trạng thái sẵn sàng khi chưa nạp thành công.

## Cách đọc chỉ số

Dice và IoU trong giao diện được tính **trên ảnh đang xem**, không phải điểm toàn bộ test. Khi cả mask dự đoán và ground truth đều rỗng, hai chỉ số được quy ước bằng 1. Thời gian là một lượt forward, đo bằng đồng hồ thực và đồng bộ CUDA khi dùng GPU; không bao gồm đọc ảnh/tiền xử lý, phụ thuộc thiết bị và có thể chậm ở lần đầu. Số tham số được đếm từ model đã nạp. GFLOPs để trống cho đến khi có phép đo đáng tin cậy. Muốn báo cáo kết quả toàn bộ test hoặc benchmark ổn định, cần chạy đánh giá riêng với cùng điều kiện cho hai model.

## API

- `GET /api/samples`: danh sách ảnh, nhãn và URL ảnh/ground truth.
- `GET /api/models`: trạng thái từng model.
- `GET /api/image?index=N` và `GET /api/ground-truth?index=N`: ảnh JPEG và mask PNG.
- `GET /api/predict?index=N&model=vit`: kết quả và chỉ số thực trên một ảnh.
- `GET /api/compare?index=N`: chạy các model sẵn sàng lần lượt trên cùng ảnh; model chưa có sẽ báo chưa sẵn sàng.

Bản web riêng tư được host tĩnh và có thể xem ảnh cùng ground truth. Model Python, checkpoint và GPU nằm trên máy chạy backend cục bộ; bản web tĩnh không tự chạy được model nếu chưa triển khai backend tương ứng. Không đưa checkpoint hoặc ảnh y tế riêng tư lên web tĩnh.

Để tạo lại bản web tĩnh từ dataset cục bộ, chạy `python -m demo.build_site`. Lệnh yêu cầu `demo/dist` chưa tồn tại và xuất 561 ảnh/mask vào đó; thư mục build được Git bỏ qua.
