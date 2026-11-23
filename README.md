# ViT và EfficientViT cho ảnh X-quang

Repo phục vụ seminar về Vision Transformer (ViT) và EfficientViT trên ảnh X-quang. Thí nghiệm hiện tại tập trung vào **segmentation nhị phân vùng u xương trên BTXRD**; ViT và EfficientViT cần được đánh giá trên cùng dữ liệu và cùng split.

## Trạng thái dự án

- Đã có ViT-B/16 từ `timm` với đầu segmentation do dự án bổ sung, loader polygon, huấn luyện, validation và đánh giá test.
- Đã chuẩn bị notebook Kaggle chạy ViT trong 20 epoch và gửi lên [Kaggle](https://www.kaggle.com/code/vtan2405/xla-btxrd-vit-segmentation-20-epochs); xem trạng thái và Output trên Kaggle để biết kết quả thực tế.
- Adapter ViT phân loại đã có, nhưng pipeline train classification và EfficientViT segmentation **chưa được triển khai**. Chưa có kết quả so sánh hai model.

ViT gốc trong [paper của Dosovitskiy và cộng sự](https://arxiv.org/abs/2010.11929) được thiết kế cho phân loại ảnh. Đầu segmentation trong repo dùng patch token của ViT, thêm convolution và nội suy thành mask; đây là phần mở rộng của dự án. Backbone dùng [timm ViT-B/16 pretrained ImageNet-21k](https://huggingface.co/timm/vit_base_patch16_224.orig_in21k), không phải code JAX/Flax gốc của tác giả.

## Dữ liệu BTXRD

Dữ liệu lấy từ [torusama/xla-seminar-uia](https://huggingface.co/datasets/torusama/xla-seminar-uia), revision `cf8b0569b18e43e7cecab412a06e60aeaf258b32`. File `btxrd_final.zip` đã tải được lưu ở `data/raw/xla-seminar-uia/`; file này và các split cục bộ được Git bỏ qua.

Ảnh trong ZIP có kích thước 224 × 224 RGB. Loader tạo mask nhị phân từ polygon, chuẩn hóa ảnh bằng mean/std `0.5` và giữ mask ở dạng 0/1. Dựa trên split gốc, đã loại các ảnh có cùng hash nhưng nhãn mâu thuẫn, sau đó loại ảnh train trùng byte với validation hoặc test:

| Tập | Bình thường | Có u | Tổng |
|---|---:|---:|---:|
| Train | 1.311 | 1.302 | 2.613 |
| Validation | 282 | 280 | 562 |
| Test | 281 | 280 | 561 |

Manifest nằm trong `data/splits/btxrd/`, gồm `train.csv`, `val.csv`, `test.csv`, `excluded.csv` và `audit.json`. Không đủ thông tin để xác nhận split tách biệt theo bệnh nhân; kiểm tra hash cũng không phát hiện bản sao đã mã hóa lại. Xem [báo cáo dữ liệu](docs/btxrd_data.md) để biết cách kiểm tra và tái tạo manifest.

FracAtlas cũng có trong dataset Hugging Face, nhưng file `fracatlas_final.zip` của bản này chỉ chứa ảnh gãy xương có annotation. BTXRD là bộ dữ liệu được chọn cho thí nghiệm hiện tại.

## Cài đặt

Khuyến nghị Python 3.10 hoặc 3.11. Cài phiên bản PyTorch/torchvision CPU hoặc CUDA phù hợp theo [hướng dẫn PyTorch](https://pytorch.org/get-started/locally/). Ví dụ PowerShell tại thư mục repo:

```powershell
$env:PYTHONUTF8 = "1"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# Install an appropriate PyTorch and torchvision build first.
python -m pip install -e ".[dev,vit]"
```

Lần đầu train với pretrained backbone, `timm` cần Internet để tải weights. Các checkpoint chỉ chứa trọng số model và metadata, không chứa trạng thái optimizer để tiếp tục train.

## Train và đánh giá ViT

Train cục bộ trên BTXRD; thư mục output phải chưa tồn tại:

```powershell
python scripts/train_vit_segmentation.py train --archive data/raw/xla-seminar-uia/btxrd_final.zip --output outputs/vit_btxrd_run1 --epochs 20 --batch-size 4 --learning-rate 0.0001
```

Lệnh tự chọn CUDA khi có GPU, nếu không dùng CPU. Giảm batch size nếu GPU thiếu bộ nhớ. `best.pt` là checkpoint có Dice validation tốt nhất; `final.pt` được lưu sau epoch cuối. `history.json` và `config.json` ghi lại quá trình và cấu hình.

Sau khi chọn model bằng validation, đánh giá trên test:

```powershell
python scripts/train_vit_segmentation.py evaluate --archive data/raw/xla-seminar-uia/btxrd_final.zip --checkpoint outputs/vit_btxrd_run1/best.pt --batch-size 4
```

Lệnh kiểm tra checksum của ZIP và báo loss, Dice và IoU foreground. Không so sánh trực tiếp metric segmentation với metric classification.

## Kaggle

Notebook [BTXRD ViT segmentation](notebooks/kaggle_btxrd_vit/btxrd_vit_segmentation.ipynb) chứa bản sao code cần thiết, tự tải đúng revision dataset và kiểm tra SHA-256. Cấu hình job dùng GPU, 20 epoch và lưu kết quả trong `/kaggle/working/btxrd_vit_20e/`. Sau khi job hoàn tất, tải `best.pt`, `final.pt`, `history.json` và `config.json` từ Output của notebook.

## Giao diện demo

Chạy `python -m demo.backend.server`, sau đó mở `http://127.0.0.1:8765`. Giao diện lấy ảnh test BTXRD, hiển thị ground truth và cho chuyển ảnh. Có thể nạp checkpoint ViT bằng `--vit-checkpoint`; EfficientViT cần được triển khai trước khi kết nối. Xem [demo/README.md](demo/README.md) để biết cấu trúc, cách chạy và ý nghĩa các chỉ số.

Checkpoint từ job ViT 20 epoch đã tải về `outputs/kaggle_vit_20e/btxrd_vit_20e/best.pt`; Dice validation bằng 0 ở cả 20 epoch. Có thể dùng file này kiểm tra luồng demo, nhưng chưa có kết quả phân vùng tốt.

## Kiểm tra và cấu trúc

```powershell
python -m xla doctor
python -m pytest -q
python -m ruff check .
```

Test dùng dữ liệu tổng hợp và không cần GPU, dataset hoặc Internet. Kiểm tra bằng model khởi tạo ngẫu nhiên là kiểm tra kỹ thuật, không phải kết quả thực nghiệm.

```text
src/xla/       Model, loader, engine và CLI
scripts/       Chuẩn bị split, train và đánh giá
notebooks/     Notebook và metadata Kaggle
data/          Dataset và manifest cục bộ, không commit
outputs/       Checkpoint và output cục bộ, không commit
docs/          Tài liệu, nguồn tham khảo và giao thức
tests/         Kiểm thử offline
paper/         Paper tham khảo của dự án
```

Xem thêm [hướng dẫn ViT segmentation](docs/vit/segmentation.md) và [danh mục nguồn](docs/references/source_registry.md).
