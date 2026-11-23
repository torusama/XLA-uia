# Train ViT cho segmentation

## Paper và mô hình

Paper ViT gốc là [An Image is Worth 16x16 Words](../../paper/1909_an_image_is_worth_16x16_words_.pdf), [Dosovitskiy và cộng sự](https://arxiv.org/abs/2010.11929), làm phân loại ảnh. Phần segmentation được bổ sung trong dự án: backbone `vit_base_patch16_224.orig_in21k` từ `timm`, bỏ token CLS, chuyển 14 × 14 patch token thành feature map, dùng hai lớp convolution và nội suy về 224 × 224. Đây là đầu segmentation đơn giản của dự án, không phải kiến trúc trong paper gốc hay TransUNet. Mask nhị phân gồm nền và vùng tổn thương; các loại u trong BTXRD được gộp thành một lớp foreground.

[Model card](https://huggingface.co/timm/vit_base_patch16_224.orig_in21k) ghi backbone pretrained ImageNet-21k và Apache-2.0. Bản PyTorch do `timm` cung cấp; repo không sao chép code ngoài. Môi trường đã kiểm tra `timm 1.0.31`, PyTorch `2.15.0+cpu`. Khi bật pretrained, `timm` tải weights theo cấu hình hiện hành và cần Internet. Revision weights chưa được cố định; cần lưu phiên bản và checksum weights riêng để tái lập hoàn toàn.

## Dữ liệu

Nguồn: [torusama/xla-seminar-uia](https://huggingface.co/datasets/torusama/xla-seminar-uia), revision `cf8b0569b18e43e7cecab412a06e60aeaf258b32`. Dataset card khai báo MIT nhưng không mô tả quy trình tạo hay điều kiện của nguồn gốc FracAtlas/BTXRD; cần kiểm tra quyền sử dụng nguồn gốc trước khi tái phân phối. Các file đã tải nằm ở `data/raw/xla-seminar-uia/` và không được commit.

| File | Train / validation / test gốc | Đặc điểm | SHA-256 |
|---|---|---|---|
| `fracatlas_final.zip` | 574 / 82 / 63 | Chỉ ảnh `Fractured`, có polygon | `3d1b3a805344bc8d10225f51a56dc2af1a4114fc717f523ec198a618128d02fe` |
| `btxrd_final.zip` | 2622 / 562 / 562 | Gồm ảnh bình thường và nhiều loại u | `59ce94244fdf997f0c68e6a378690a4359e6c411f7fda7ac2f3994489637884b` |

Ảnh đã resize về 224 × 224; polygon có tọa độ trên ảnh đó. Loader đọc trực tiếp ZIP, tô polygon, dùng mask rỗng cho `Normal`, chuyển ảnh RGB và chuẩn hóa mean/std 0.5. Ảnh không phải `Normal` mà thiếu polygon sẽ gây lỗi.

Loader loại ảnh có hash giống nhau nhưng nhãn khác nhau ở mọi split, sau đó loại ảnh train trùng hash với validation hoặc test; nếu validation và test trùng hash thì dừng. BTXRD còn 2.613 train, 562 validation và 561 test. Kiểm tra này không phát hiện ảnh giống nhau đã mã hóa khác hay bệnh nhân trùng giữa split; xem [báo cáo BTXRD](../btxrd_data.md). Không dùng test để chọn model.

## Cài đặt và chạy

Trong PowerShell tại thư mục dự án:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,vit]"
```

Để dùng GPU, cài PyTorch/torchvision phù hợp CUDA theo [hướng dẫn PyTorch](https://pytorch.org/get-started/locally/). Nếu chưa có ZIP, tải đúng revision:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='torusama/xla-seminar-uia', repo_type='dataset', revision='cf8b0569b18e43e7cecab412a06e60aeaf258b32', local_dir='data/raw/xla-seminar-uia')"
```

Train BTXRD; dùng thư mục output chưa tồn tại để tránh ghi đè:

```powershell
python scripts/train_vit_segmentation.py train --archive data/raw/xla-seminar-uia/btxrd_final.zip --output outputs/vit_btxrd_run1 --epochs 20 --batch-size 4 --learning-rate 0.0001
```

Lệnh chạy trên máy đang thực thi, dùng GPU nếu có. Lần đầu cần Internet để tải pretrained weights; `--from-scratch` dùng khởi tạo ngẫu nhiên. Giảm `--batch-size` nếu thiếu bộ nhớ GPU. Output gồm `config.json`, `history.json`, `best.pt` chọn theo Dice foreground trên validation và `final.pt` tại epoch cuối. Loss là BCE với trọng số pixel foreground mặc định 10 cộng soft Dice tính theo batch; có thể đổi bằng `--positive-weight`. Đây là cấu hình thử nghiệm để xử lý mất cân bằng pixel, chưa đảm bảo model sẽ hội tụ. Checkpoint chứa trọng số model và metadata, không chứa optimizer để tiếp tục train.

Đánh giá test sau khi chốt cấu hình:

```powershell
python scripts/train_vit_segmentation.py evaluate --archive data/raw/xla-seminar-uia/btxrd_final.zip --checkpoint outputs/vit_btxrd_run1/best.pt --batch-size 4
```

Lệnh kiểm tra SHA-256 của ZIP và in loss, Dice, IoU foreground toàn bộ pixel. Không so sánh trực tiếp metric segmentation với classification. Mask 224 × 224 có thể làm mất chi tiết nhỏ; pipeline chưa được xác thực lâm sàng.

## Notebook Kaggle

Notebook [BTXRD ViT 20 epoch](../../notebooks/kaggle_btxrd_vit/btxrd_vit_segmentation.ipynb) chứa bản sao code, tự tải revision dataset và kiểm tra SHA-256. Job dùng GPU, Internet, batch 4, 20 epoch và lưu output vào `/kaggle/working/btxrd_vit_20e`. Tải `best.pt` hoặc `final.pt` từ Output sau khi job hoàn tất.

Kiểm tra offline bằng `python -m pytest -q` và `python -m ruff check .`; test dùng dữ liệu tổng hợp, không cần dataset, GPU hay Internet.
