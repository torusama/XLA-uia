# CODEX TASK C4.2 — ViT Classification (Python + PyTorch)

> **ĐẶC TẢ GIAO VIỆC CHO CODEX — THỰC HIỆN SAU PHASE 0**  
> **Task:** C4.2 — Nghiên cứu và cài đặt lại quy trình ViT Classification trên ảnh X-quang FracAtlas.  
> **Stack thống nhất cả đồ án:** **Python + PyTorch**. Task EfficientViT (Han Cai et al., ICCV 2023) ở giai đoạn sau cũng chạy PyTorch.  
> **Nguyên tắc:** Dùng **implementation và pretrained weights có sẵn, nguồn gốc xác thực**. **KHÔNG viết lại ViT backbone, patch embedding, attention, transformer encoder từ đầu.** Chỉ viết phần tích hợp dataset, cấu hình, orchestration, huấn luyện/đánh giá khi thật sự cần.  
> **Đây là spec cho Codex tự code**, KHÔNG phải mô tả một implementation hiện đã hoàn thành.

---

## 0. QUYẾT ĐỊNH KỸ THUẬT QUAN TRỌNG NHẤT

1. **Python là ngôn ngữ lập trình; JAX/Flax và PyTorch đều là framework Python.** Tuy nhiên, **toàn bộ project này chọn PyTorch làm framework chạy chính**, để ViT và EfficientViT dùng chung data, training, metrics, checkpoint và benchmark.
2. **Paper ViT core:** *An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale* — Dosovitskiy et al., ICLR 2021. **Repo chính thức của Google:** https://github.com/google-research/vision_transformer. Repo Google cung cấp implementation **JAX/Flax**, **không phải một official PyTorch training implementation**. Vì vậy **không** nhập JAX/Flax làm runtime chính của đồ án.
3. **Nguồn model PyTorch để chạy:** [timm](https://github.com/huggingface/pytorch-image-models) — `vit_base_patch16_224.orig_in21k` / [model card](https://huggingface.co/timm/vit_base_patch16_224.orig_in21k). Model card xác nhận weights được các tác giả ViT pretrain bằng JAX trên ImageNet-21k, **Ross Wightman đã port sang PyTorch**. **Đây là PyTorch port, không được mô tả là code PyTorch chính thức do tác giả ViT viết.**
4. **EfficientViT cho task sau:** repo **MIT Han Lab** https://github.com/mit-han-lab/efficientvit, ứng dụng classification https://github.com/mit-han-lab/efficientvit/tree/master/applications/efficientvit_cls — đã dùng PyTorch. **Không triển khai EfficientViT trong task C4.2**, nhưng không làm hỏng interface chung cho lần sau.
5. Có thể **đọc/clone repo Google để nghiên cứu thuật toán và đối chiếu cấu trúc**, nhưng **forward/train thực tế phải chạy bằng PyTorch/timm**, tránh hai framework khác nhau trong cùng pipeline.
6. **Không dùng EViT (Expediting Vision Transformers, token-pruning) để thay EfficientViT của Han Cai.** Đây là hai họ kiến trúc khác nhau.

### Độ trung thực về nguồn code (BẮT BUỘC ghi trong báo cáo)

| Hạng mục | Nguồn đúng | Trách nhiệm nhóm |
|---|---|---|
| Ý tưởng, công thức, kiến trúc ViT | Paper Dosovitskiy 2021 + repo Google JAX/Flax | Đọc, hiểu, mô tả chính xác |
| ViT-B/16 chạy trong PyTorch | `timm` / mã PyTorch port của Ross Wightman | Cấu hình, tái sử dụng, không viết lại backbone |
| Trọng số pretrained | Checkpoint ViT ImageNet-21k port từ tác giả, theo model card | Ghi model ID, revision/checksum nếu xác minh được |
| FracAtlas, train/val/test, metric, CLI | Code nhóm phát triển quanh model có sẵn | Implement + test |
| EfficientViT (giai đoạn kế tiếp) | MIT Han Lab, ICCV 2023 | Tích hợp theo cùng protocol |

**Không tồn tại một phương án vừa “chạy nguyên xi official Google ViT source” vừa “chạy ViT bằng PyTorch”** nếu không chuyển framework/port mã. Ở đây đã chốt **ưu tiên hệ thống dùng chung PyTorch, tận dụng bản port đáng tin cậy** và giữ Google source làm chuẩn nghiên cứu.

---

## 1. NHIỆM VỤ CODEX PHẢI LÀM NGAY

Trước tiên đọc `README_CODEX_SETUP_XLA.md`, `README.md`, `AGENTS.md`, `Research.pdf` (nếu có trong workspace), kiểm tra repo thật bằng `git status` và liệt kê code đã tồn tại. **Không tự tạo project mới, không xóa hoặc ghi đè code Phase 0, không thay đường dẫn/config tùy tiện.**

**Hoàn thành C4.2 bằng những bước sau:**

1. **Audit nguồn:** đọc paper ViT, repo Google (đặc biệt `vit_jax/models_vit.py`, config), repo `timm` (đặc biệt implementation `VisionTransformer`) và model card của checkpoint gốc. Tạo `docs/vit/source_review.md`, `docs/vit/code_walkthrough.md` với link đến code, mô tả từng tensor và điểm tương đồng/khác biệt ở PyTorch port. Không sao chép file gốc toàn bộ vào source nhóm nếu không cần.
2. **Setup Python/PyTorch:** bổ sung dependencies cho task, giữ `pyproject.toml`/cấu hình Phase 0; tự kiểm tra version tương thích Windows/Colab/Linux, không tự đóng đinh một wheel CUDA cho mọi máy.
3. **Data audit FracAtlas:** kiểm tra **cấu trúc dataset thật**, trường metadata, loại nhãn, ảnh còn/thiếu, nhãn 0/1; tạo manifest `train.csv`, `val.csv`, `test.csv` cố định. Không giả định có patient ID khi chưa xác minh. Không tải dataset lớn âm thầm.
4. **ViT model adapter:** dùng `timm.create_model` hoặc API tương đương của `timm` để load **ViT-B/16 original pretrained** và thay classifier thành **2 lớp**. Giữ backbone và pretrained weight; không viết lớp `Attention`, `TransformerEncoder` hay `VisionTransformer` riêng.
5. **Training:** fine-tune trên FracAtlas sử dụng pipeline PyTorch, tái sử dụng `timm` utilities/training code phù hợp khi có thể. Xử lý config, optimizer, loss, scheduler, logging, validation, resume, checkpoint `.pth`/`.pt`, seed.
6. **Evaluate & predict:** load checkpoint thật, tính Accuracy/Precision/Recall/F1 và confusion matrix trên hold-out test; tạo output CSV/JSON và khả năng inference ảnh mới.
7. **Tests và docs:** có CPU unit/smoke tests không đòi dataset thật và end-to-end integration tests có điều kiện; cập nhật `README.md` với lệnh chạy **thật** từ source hiện có. Các câu lệnh ở tài liệu này là **hợp đồng để Codex implement**, chưa phải tuyên bố đang chạy được.
8. Chạy các test và smoke test trên môi trường thực; báo PASS/FAIL/BLOCKED minh bạch. **Không bịa checkpoint, kết quả, thông số benchmark hoặc việc đã train nếu chưa train.**

**KHÔNG làm trong task này:** train EfficientViT, implement segmentation, tạo ablation giả, biến input FracAtlas thành ảnh mask, thay task sang BTXRD nếu chưa được yêu cầu.

---

## 2. PAPER / CODE / WEIGHTS BẮT BUỘC

| Tài liệu | Link | Vai trò |
|---|---|---|
| **ViT paper gốc (CORE)** | https://arxiv.org/abs/2010.11929 | Lý thuyết gốc |
| **Google official repo (JAX/Flax)** | https://github.com/google-research/vision_transformer | Đối chiếu thuật toán, kiến trúc, nguồn checkpoint |
| **PyTorch implementation (`timm`)** | https://github.com/huggingface/pytorch-image-models | Runtime chính cho ViT và pretrained port |
| **Original pretrained ViT-B/16 port** | https://huggingface.co/timm/vit_base_patch16_224.orig_in21k | Model ID và provenance của checkpoint gốc |
| **Tài liệu timm model API** | https://huggingface.co/docs/timm/main/reference/models | `create_model`, `num_classes`, `reset_classifier` |
| **FracAtlas** | https://github.com/XLR8-07/FracAtlas | X-ray dataset + annotation/metadata |
| **FracAtlas paper** | https://www.nature.com/articles/s41597-023-02432-4 | Dataset protocol/limitations |
| **EfficientViT paper core (CHO TASK SAU)** | https://openaccess.thecvf.com/content/ICCV2023/html/Cai_EfficientViT_Lightweight_Multi-Scale_Attention_for_High-Resolution_Dense_Prediction_ICCV_2023_paper.html | Phân biệt đúng EfficientViT |
| **MIT EfficientViT source (CHO TASK SAU)** | https://github.com/mit-han-lab/efficientvit | Chừa interface PyTorch tương thích |

**Provenance:** Codex phải ghi URL, version/commit thực tế, license đã kiểm tra và checksum/revision của checkpoint nếu truy xuất được tại `docs/references/source_registry.md`; không tự điền SHA tưởng tượng. Lưu `pip freeze`, torch/timm versions, OS và device trong mỗi experiment.

**Vì muốn nghiên cứu core paper:** `docs/vit/code_walkthrough.md` phải giải thích được: ảnh `[B,3,224,224]` → patch embedding (patch 16) → 196 patch tokens + CLS = 197 tokens → positional embedding → 12 Transformer blocks (ViT-B/16, 12 heads, hidden dim 768) → CLS embedding → classification head `[B,2]`. Kiểm tra các thông số thực tế bằng config/model introspection, **không chỉ chép bảng trong paper**.

---

## 3. PHẠM VI BÀI TOÁN / PROTOCOL CHUNG CHO EFFICIENTVIT SAU NÀY

- **Task:** binary **image classification**, **không** phải detection hay segmentation.
- **Dataset chính:** FracAtlas; ảnh X-quang có nhãn fractured / non-fractured, nhãn phải đọc từ metadata thật sau khi audit.
- **Input:** một ảnh, `sample_id`; RGB 3-channel cho checkpoint (chuyển grayscale sang RGB nếu có, không coi đó là ba kênh thông tin độc lập).
- **Ground truth:** `0 = non_fractured`, `1 = fractured` (khẳng định mapping sau khi đối chiếu dataset thật; ghi vào file JSON/versioned).
- **Output:** logits `[B,2]`, softmax probabilities, predicted class.
- **Loss:** `torch.nn.CrossEntropyLoss` cho hai lớp loại trừ nhau. Nếu dùng class weight phải tính **chỉ từ train** và ghi rõ.
- **ViT input:** mặc định 224×224 theo checkpoint đã chọn; không tự đổi kích thước rồi giả định vị trí embedding tương thích.
- **Metrics:** accuracy, precision/recall/F1 theo lớp, macro-F1, weighted-F1, confusion matrix, optional ROC-AUC với positive probability khi hợp lệ.
- **Data leakage:** train/val/test cố định, chung cho ViT và EfficientViT. Nếu có patient/group ID đủ tin cậy, split theo group; nếu không có, dùng split có stratification và **báo giới hạn**; kiểm tra trùng file/ảnh gần trùng. Không phán rằng chia image là patient-level.
- **Model selection:** dùng **validation**; không dùng test để tune hyperparameters, early stopping, threshold hoặc normalization. Final test chỉ sau khi chọn checkpoint.
- **Preprocessing:** xác minh size, aspect ratio, EXIF orientation, grayscale format, bit depth; không làm mất vùng gãy do crop một cách không kiểm soát. Chỉ train augmentation; val/test deterministic. Đọc `timm.data.resolve_model_data_config` để biết normalization gốc, không đoán mean/std.
- **Để so sánh tương lai:** giữ chung manifest, image geometry policy đã công bố, metrics, seed, precision, device, batch/inference protocol. Phân biệt **matched-resolution comparison** với **model-native-resolution comparison**, nếu dùng cả hai.

Lưu ý: nhiều ảnh âm tính có thể chiếm đa số; **không được chỉ báo Accuracy mà bỏ Precision/Recall/F1 của lớp fractured**.

---

## 4. CẤU TRÚC THƯ MỤC — BỔ SUNG TRÊN REPO PHASE 0, KHÔNG DỰNG LẠI

```text
xla-vit-efficientvit/
├── README.md                                  # bổ sung cách chạy C4.2
├── README_CODEX_SETUP_XLA.md                   # giữ nguyên nếu có
├── pyproject.toml                              # thêm optional dependencies/train deps
├── configs/
│   ├── datasets/fracatlas.yaml                # cập nhật theo metadata kiểm chứng
│   ├── models/vit_b16.yaml                    # timm model ID, num_classes, pretrained
│   └── experiments/vit_fracatlas_c4_2.yaml    # split, seed, train config, checkpoint
├── src/xla/
│   ├── data/
│   │   ├── fracatlas.py                       # dataset adapter; output image/label/id
│   │   ├── splitting.py                       # fixed train/val/test manifests
│   │   ├── inspect.py                         # thống kê dataset
│   │   └── preprocess.py                      # train/eval transforms
│   ├── models/
│   │   ├── vit.py                             # thin timm adapter, KHÔNG tự viết ViT
│   │   └── registry.py                        # giữ tương thích model registry Phase 0
│   ├── engine/
│   │   ├── trainer.py                         # shared torch train logic
│   │   ├── evaluator.py                       # eval logic
│   │   ├── inferencer.py                      # inference shared
│   │   └── checkpoints.py                     # torch ckpt + metadata
│   └── metrics/
│       └── classification.py                  # standard metric implementation
├── scripts/
│   ├── inspect_fracatlas.py                   # inspect + sanity checks
│   ├── prepare_fracatlas.py                   # create/verify fixed manifests
│   ├── train_vit.py                           # CLI train: config-driven
│   ├── evaluate_vit.py                        # CLI final test from checkpoint
│   └── predict_vit.py                         # CLI inference 1/multiple images
├── tests/
│   ├── test_vit_adapter.py                    # shape + head + pretrained load path
│   ├── test_fracatlas_labels.py               # labels and class order
│   ├── test_split_integrity.py                # no overlaps / group leakage checks
│   ├── test_checkpoint_roundtrip.py           # save + reload + same logits
│   ├── test_metrics_classification.py         # known truth -> known metrics
│   └── test_vit_smoke.py                      # tiny CPU forward/1 train step
├── data/
│   ├── raw/fracatlas/                         # dataset thật, gitignored
│   └── splits/fracatlas/                      # train.csv, val.csv, test.csv
├── docs/
│   ├── tasks/C4_2_VIT_CLASSIFICATION.md      # copy/move bản đặc tả này
│   ├── vit/source_review.md                   # Google -> timm port provenance
│   ├── vit/code_walkthrough.md                # full ViT path + tensor shapes
│   ├── vit/data_protocol.md                   # verified labels, split, transforms
│   ├── vit/training_protocol.md               # optimizer/scheduler/loss/validation
│   └── vit/results.md                         # ghi kết quả THỰC hoặc BLOCKED
├── outputs/
│   └── vit/                                   # checkpoint/metrics/predictions/figures
└── third_party/
    └── README.md                              # nguồn liên kết; chỉ vendor khi cần
```

**Cấm** tự tạo repo JAX/Flax thứ hai nằm song song để train ViT, cấm tự thay `.pth` bằng `.npz` JAX. Không copy `timm` source thành `src/xla/models/vit.py`: adapter phải import và gọi thư viện; giữ license/citation khi vendor third-party code nếu thật sự cần.

---

## 5. PYTORCH MODEL LOADING — VÍ DỤ HƯỚNG DẪN CODEX, KHÔNG PHẢI CODE ĐÃ CHẠY

```python
import torch
import timm

MODEL_NAME = "vit_base_patch16_224.orig_in21k"
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=2)
model.eval()

with torch.inference_mode():
    logits = model(torch.zeros(1, 3, 224, 224))
assert logits.shape == (1, 2)
```

- `pretrained=True` cần tải checkpoint khi chưa cache; **không** nói là đã tải/trained nếu chưa có môi trường/network. Nếu model ID resolve thất bại do phiên bản `timm`, kiểm tra **đúng model registry / model card và pin version**; báo lỗi cụ thể, không chuyển sang random init.
- Vì `orig_in21k` là **backbone pretrained không có classification head** trong model card, `num_classes=2` tạo **head mới ngẫu nhiên**, còn backbone phải nhận weights pretrained; kiểm tra missing/unexpected keys hoặc consistency bằng log, không khẳng định head cũng pretrained.
- `timm` model default config chứa **preprocessing / normalization**; trích xuất từ model đã load. Train và inference phải lưu/chạy lại chính xác transform config.
- Viết smoke test kiểm tra logits hữu hạn, shape `[B,2]`, gradients hữu hạn cho một step; test không tải pretrained có thể dùng `pretrained=False` chỉ khi ghi rõ **UNIT TEST RANDOM INIT**, tuyệt đối không đưa kết quả đó vào bảng thực nghiệm.
- `timm` có các script train/validation và utility đang được duy trì; Codex phải **nghiên cứu và tái sử dụng** phần phù hợp. Nếu cần custom `engine/trainer.py` để giữ protocol chung với EfficientViT sau này, chỉ viết orchestration dùng `torch`/`timm`, **không duplicate implementation của backbone**.

**Tham khảo tùy chọn:** Hugging Face `transformers` có `AutoImageProcessor`/`ViTForImageClassification`, nhưng task này **chốt `timm` làm runtime ViT chính**. Không dùng hai framework wrapper khác nhau cho cùng một experiment rồi trộn kết quả.

---

## 6. CÁCH CODEX SETUP MÔI TRƯỜNG

Ưu tiên **Python 3.11**, nếu constraints của repo MIT khi sang C4.3 không tương thích thì có thể dùng **Python 3.10** đồng nhất cả hai; kiểm tra thực tế trước khi khóa version. **Không ép JAX, Flax, TensorFlow, TFDS hay TensorRT** trong C4.2.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Cài PyTorch CPU/CUDA từ hướng dẫn https://pytorch.org/get-started/locally/
python -m pip install -e ".[dev]"
# Codex phải bổ sung dependencies C4.2 (timm, Pillow, numpy, pandas,
# scikit-learn, PyYAML, matplotlib...) vào project trước khi cài.
python -m xla doctor
python -m pytest -q
```

### Linux / Colab

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m xla doctor
python -m pytest -q
```

Codex phải cung cấp **lệnh cài PyTorch tương ứng môi trường đã phát hiện**, tránh hardcode CUDA. Trên Colab/VS Code, chỉ dẫn `python -m` từ **repo root** và dùng đường dẫn config tương đối.

### Config yêu cầu (Codex tự tạo và validate)

```yaml
experiment_name: vit_fracatlas_c4_2
seed: 42
model:
  provider: timm
  name: vit_base_patch16_224.orig_in21k
  pretrained: true
  num_classes: 2
  image_size: 224
dataset:
  name: fracatlas
  root: data/raw/fracatlas
  split_dir: data/splits/fracatlas
  labels: [non_fractured, fractured]
training:
  device: auto
  epochs: 20       # starting-point config only; tune by validation
  batch_size: 8    # adapt to VRAM, record actual value
  optimizer: adamw
  learning_rate: 0.00003
  weight_decay: 0.01
  early_stopping_metric: val_macro_f1
output_dir: outputs/vit
```

**Các con số epoch/LR/batch là cấu hình khởi đầu, không phải hyperparameters được paper ViT chứng minh tối ưu cho FracAtlas.** Codex cần verify YAML schema, config snapshot và sự phù hợp với GPU trước khi train.

---

## 7. DATASET PREPARATION VÀ TEST SPLIT

**Không đoán file layout của FracAtlas.** Codex đọc dataset repo/metadata thực tế trước khi viết parser, sau đó tài liệu hóa schema tại `docs/vit/data_protocol.md`.

Yêu cầu:

1. Chỉ rõ đường dẫn ảnh, metadata, nhãn gốc, cách chuyển sang `0/1`, những ảnh bỏ qua và nguyên nhân. Nếu dữ liệu đã tồn tại, **không download lại**.
2. Manifest chứa tối thiểu: `sample_id`, `image_path`, `label`, `split`; thêm `patient_id/group_id` **chỉ khi nguồn xác minh có**.
3. Sử dụng **train/val/test cố định** cho hai model. Kiểm tra unique IDs, không sample overlap, không file-hash overlap; group overlap nếu có group ID.
4. Nếu dataset đã có benchmark split chính thức, ưu tiên giữ và làm rõ cách tách val. Nếu không có, tạo stratified split có seed, **đưa tỷ lệ split vào config** và lưu kết quả/version/hash.
5. Áp dụng augmentation **chỉ trên train**; mọi phép lật/xoay/crop/contrast cho ảnh y khoa cần xem lại tính hợp lý giải phẫu và bbox lesion. **Không mặc định dùng `RandomResizedCrop` mạnh** có thể cắt mất gãy xương.
6. Chuyển định dạng ảnh cho model phải không làm mất nhãn và không tạo artifact lẫn vào val/test; ghi nhận aspect ratio và cách resize/pad.
7. Phát hiện ảnh lỗi, dữ liệu trùng, nhãn không hợp lệ trước khi train. Không thay thế nhãn thiếu bằng suy đoán.
8. Xuất báo cáo class distribution cho từng split; dùng macro-F1, confusion matrix và recall fractured làm metric cốt lõi.

---

## 8. TRAIN / EVALUATE / PREDICT — CONTRACT CHO CODEX IMPLEMENT

Codex cần làm cho **những lệnh dưới đây chạy thật**. Nếu cấu trúc CLI Phase 0 đang có khác, có thể giữ CLI thống nhất `python -m xla ...`; phải ghi mapping lệnh rõ trong README chung.

```bash
# Chỉ kiểm tra metadata/labels, không cần GPU
python scripts/inspect_fracatlas.py --config configs/datasets/fracatlas.yaml

# Tạo/sử dụng lại train/val/test manifests; in overlap audit
python scripts/prepare_fracatlas.py --config configs/experiments/vit_fracatlas_c4_2.yaml

# Train từ pretrained ViT-B/16, đánh giá validation theo epoch
python scripts/train_vit.py --config configs/experiments/vit_fracatlas_c4_2.yaml

# Resume checkpoint nếu bị dừng
python scripts/train_vit.py --config configs/experiments/vit_fracatlas_c4_2.yaml --resume outputs/vit/<run_id>/checkpoints/last.pth

# Đánh giá test một lần từ best checkpoint trên validation
python scripts/evaluate_vit.py --config configs/experiments/vit_fracatlas_c4_2.yaml --checkpoint outputs/vit/<run_id>/checkpoints/best.pth --split test

# Predict ảnh mới, không đòi ground-truth
python scripts/predict_vit.py --checkpoint outputs/vit/<run_id>/checkpoints/best.pth --image path/to/image.jpg
```

**Điều kiện training thật:** raw data thật + pretrained weights + thiết bị đáp ứng. Nếu không đủ, chạy smoke test và báo `BLOCKED`, không tạo loss/accuracy giả. Test unit tự sinh có thể dùng random tensors nhưng phải tách khỏi mọi báo cáo thí nghiệm.

**Checkpoint PyTorch** chứa tối thiểu: `state_dict`, `optimizer_state`/`scheduler_state` nếu resume, `epoch`, `best_val_metric`, `model_name`, `label_map`, `config`, `seed`, `preprocessing`, `split_hashes`, package versions. Tất cả dùng `.pth`/`.pt`, `map_location` khi load, từ đường dẫn tin cậy. Reload phải cho cùng logits (khi `eval()` và cùng inputs).

**Kết quả test** phải lưu: `metrics_test.json`, `predictions_test.csv` gồm `sample_id/label/score/pred`, `confusion_matrix.png`, training history, environment/config snapshot, file hash/commit source nếu có. Không report số liệu nếu chưa chạy.

**Training log:** epoch/step, train loss, val loss, val accuracy/macro-F1, LR, GPU memory (nếu GPU), thời gian; nhớ có log ngay khi bắt đầu, không im lặng nhiều epoch.

---

## 9. TESTS BẮT BUỘC / DEFINITION OF DONE

- [ ] C4.2 chạy **Python + PyTorch**, không đòi JAX/Flax.
- [ ] `docs/vit/source_review.md` chứng minh đã đọc **ViT paper + Google source + PyTorch port**. Nêu rõ port không phải source tác giả viết bằng PyTorch.
- [ ] Source/model registry thực sự gọi `timm`, **không** tự viết ViT backbone.
- [ ] Load checkpoint pretrained thành công, log rõ model ID, source revision/version, classifier 2 lớp; không nhầm random init với pretrained.
- [ ] Unit test trên CPU kiểm tra input shape, logits `[B,2]`, finite gradients, label consistency, metrics đúng trên toy examples, checkpoint round-trip.
- [ ] Dataset adapter đọc **nhãn thật** và tạo fixed train/val/test manifests; audit duplicate/leakage, thống kê class distributions.
- [ ] Val chọn best checkpoint; test không bị dùng để select model/hyperparams.
- [ ] Lệnh train/evaluate/predict có `--help`, error messages rõ ràng, không hardcode path cá nhân.
- [ ] Kết quả của full fine-tune + final test chỉ đánh **PASS** khi đã chạy trên FracAtlas thật. Khi thiếu data/GPU/network phải ghi `BLOCKED: <reason>`.
- [ ] Code chủ yếu reuse PyTorch framework/timm implementation, chỉ viết adapter/data/training orchestration cần thiết.
- [ ] `python -m pytest -q`, `python -m ruff check .`, `python -m xla doctor` và CPU smoke-test được thực thi, ghi output tóm tắt.
- [ ] Không sửa/implement nhánh EfficientViT, segmentation, BTXRD trong task này; đảm bảo về sau có thể dùng **chung split + common metrics + torch checkpoints**.
- [ ] Tạo `docs/vit/results.md` báo chính xác PASS/BLOCKED và các script/command đã chạy, không bịa số liệu.

---

## 10. BẮT BUỘC ĐỌC CODE ĐỂ THUYẾT TRÌNH THUẬT TOÁN

Nội dung `docs/vit/code_walkthrough.md` cần gắn `file/function` của **repo Google (JAX) và `timm` (PyTorch)**, trả lời được:

1. Ảnh thành patch tokens và CLS token thế nào? Với 224 và patch 16, vì sao có **196 + 1 = 197 tokens**?
2. Learned positional embedding nằm ở đâu? Khi thay resolution, chuyện gì xảy ra với positional embedding?
3. Query, Key, Value và công thức **softmax(QKᵀ / √dₖ)V** nằm ở module nào của PyTorch port? Kích thước tensor trước/sau attention?
4. Multi-head attention, LayerNorm, residual, MLP/GELU và 12 encoder blocks được khởi tạo thế nào?
5. Vì sao self-attention ở N tokens có hạng chi phí **O(N²d)** và bộ nhớ attention **O(N²)** (trừ các kernel tối ưu fused attention)? Nêu rõ đây là phân tích thuật toán, không thay thế benchmark thật.
6. ViT-B/16 dùng output CLS như thế nào? Classifier 2 lớp mới có shape nào? Phần pretrained thực sự được load là gì?
7. Quy trình `forward → loss → backward → optimizer.step → validation → checkpoint` chạy ở file/hàm nào? Đâu là source upstream, đâu là code nhóm tích hợp?
8. Loss CrossEntropy khác gì Accuracy/F1? Vì sao tối ưu loss không tương đương trực tiếp tối ưu macro-F1?
9. Những rủi ro riêng của X-quang: biến dạng tỷ lệ, mất dấu gãy do crop, class imbalance, duplicates, domain shift từ ImageNet sang X-ray?
10. Những thông số nào khi so EfficientViT **phải giống nhau** (task/dataset/split/device/protocol) và những thông số nào cần báo riêng (native resolution, pretrained source, number of params, MACs, latency)?

**Không được viết “nhóm phát minh / implement from scratch” cho attention hay ViT.** Nhóm đang **nghiên cứu, tái sử dụng implementation và fine-tune** theo hướng dẫn đồ án.

---

## 11. CÁCH CODEX BÁO CÁO SAU KHI CODE XONG

Trong phản hồi cuối, trình bày ngắn gọn:

1. Thay đổi file, những module reuse, commit/version upstream.
2. Câu lệnh setup và chạy đã **kiểm tra thực tế**; phiên bản Python/Torch/timm và CPU/GPU.
3. Kiểm tra dataset: có/không có, kích thước, nhãn, splits và leakage audit nếu làm được.
4. Smoke tests, unit tests, lint: PASS/FAIL (đính kèm thông báo lỗi khi fail).
5. Full fine-tune/test: hoàn thành, hoặc `BLOCKED` và lý do cụ thể.
6. Hướng tích hợp với task C4.3 EfficientViT: **chỉ ghi interface/chú ý**, không tự implement.

**Mục đích cuối cùng:** Một baseline ViT Classification bằng **Python/PyTorch**, tái sử dụng **port ViT gốc đã công bố**, có thể train/evaluate/predict và có protocol đủ sạch để sang EfficientViT so sánh trên cùng FracAtlas.
