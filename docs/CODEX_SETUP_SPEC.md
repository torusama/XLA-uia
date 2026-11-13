# XLA — Vision Transformer vs EfficientViT

> **PROJECT BLUEPRINT / SETUP SPEC FOR CODEX — PHASE 0**  
> Đây là **bản hướng dẫn để Codex dựng nền tảng dự án từ đầu**, **không phải** mô tả một codebase đã hoàn thành.  
> Ngôn ngữ tài liệu: Tiếng Việt. Tên file, API, biến, comment trong source code: Tiếng Anh.

---

## 0. Lệnh thực hiện dành cho Codex — ĐỌC PHẦN NÀY TRƯỚC

**Nhiệm vụ:** Sau khi đọc README này, hãy tạo cấu trúc thư mục và bộ setup kỹ thuật ban đầu cho đồ án Xử lý ảnh / Computer Vision nghiên cứu **Vision Transformer (ViT)** và **EfficientViT (Han Cai et al., ICCV 2023)**. Mục tiêu của lần chạy này chỉ là **PHASE 0 — scaffold & environment**, để các thành viên có thể bắt đầu implement từng thuật toán sau đó.

**Quy tắc bắt buộc:**

1. Trước khi tạo file, **đọc cây thư mục hiện tại**. Nếu repo đã có code, **không xóa/ghi đè các file có giá trị**; đề xuất hoặc thực hiện thay đổi tối thiểu, giữ lịch sử và tránh sửa code không liên quan.
2. **Tạo thư mục + file cấu hình + khung module + CLI kiểm tra môi trường + unit tests tối thiểu** như mô tả ở dưới. Những module cho training, segmentation, benchmark có thể là interface/TODO có docstring rõ ràng.
3. **KHÔNG** tự triển khai full ViT/EfficientViT từ đầu trong Phase 0. **KHÔNG** làm pipeline huấn luyện hoàn chỉnh, **KHÔNG** báo kết quả Accuracy/FLOPs/latency chưa đo, **KHÔNG** tạo checkpoint giả.
4. **KHÔNG** tự tải dataset, model weights, ZIP hàng chục GB, hay clone repo lớn. Đặt các link và hướng dẫn tải ở tài liệu; chỉ clone khi người dùng yêu cầu ở giai đoạn sau.
5. **KHÔNG** chép source bên thứ ba vào project khi chưa xem license; ưu tiên Python package/adapter, hoặc lưu link + commit gốc ở `docs/references/source_registry.md`.
6. Bảo đảm chạy được trên **Windows + VS Code**; cho phép **Linux/Google Colab** về sau. Không phụ thuộc `bash`, `make` hay đường dẫn tuyệt đối để chạy các thao tác cơ bản.
7. Khi thiếu dataset, GPU hay checkpoint, chương trình phải **thông báo thiếu tài nguyên rõ ràng**, không crash khó hiểu và không âm thầm dùng dữ liệu mô phỏng để thay thế thí nghiệm thật.
8. Sau khi dựng scaffold, thực sự chạy các bước kiểm tra **Definition of Done** ở Mục 11, sửa lỗi cho đến khi pass (trong phạm vi môi trường đang có), rồi tóm tắt **file nào được tạo, test nào pass/fail và việc gì chưa làm**.
9. Nếu có sự khác nhau giữa yêu cầu, kiến trúc và tài liệu gốc: ưu tiên các ràng buộc ở README này, đồng thời ghi quyết định/điểm chưa rõ trong `docs/decisions.md` thay vì tự khẳng định.

**Chỉ làm Phase 0 trong lần này.** Các Phase 1–4 phía dưới là roadmap và hợp đồng giao tiếp giữa các module, **không phải** yêu cầu triển khai hết ngay lập tức.

---

## 1. Tổng quan đề tài và nguồn yêu cầu

**Tên đề tài tạm thời:** *Khảo sát và so sánh Vision Transformer và EfficientViT về độ chính xác và hiệu quả tính toán trên ảnh y khoa.*

**Phạm vi lấy từ tài liệu của nhóm:**

- `Document.pdf`: hướng dẫn của giảng viên về **problem statement → related works → phương pháp → cài đặt/thử nghiệm → đánh giá**. Giảng viên nhấn mạnh input/output/ground truth/loss, tiêu chí chính xác và độ phức tạp, tránh đưa tên mô hình cụ thể vào framework tổng quát của bài toán.
- `Research.pdf`: phân nhánh **ViT** và **EfficientViT**, các paper liên quan, FracAtlas/BTXRD, các thí nghiệm Attention Scaling, Classification và Segmentation, cùng định hướng tổ chức source.
- `Paper.zip`: tập các PDF nghiên cứu ViT, DeiT, Swin, TransUNet, EfficientViT, SegFormer, SegNeXt, TopFormer, SeaFormer, FracAtlas, BTXRD... (tài liệu tham khảo, **không phải source code**).
- `LT.zip`: tài liệu môn học / bài giảng (không phải mã nguồn project).

### Mục tiêu nghiên cứu

1. Giải thích nguyên lý **patch embedding, position embedding, transformer encoder, multi-head self-attention** trong ViT.
2. Giải thích **ReLU-based linear attention + multi-scale token aggregation + depthwise convolution** trong EfficientViT ICCV 2023.
3. Có quy trình thí nghiệm **cùng bài toán, cùng tập dữ liệu/split và điều kiện đo** để đối chiếu Accuracy/F1 với MACs/FLOPs, latency, memory, parameters.
4. Dựng được 2 nhánh nghiên cứu độc lập:
   - **Track A (ưu tiên): Image classification** — ViT-B/16 vs EfficientViT classification trên **FracAtlas**; sau đó tùy chọn BTXRD.
   - **Track B (mở rộng): Semantic segmentation** — mô hình segmentation phù hợp với ViT (ví dụ TransUNet) vs EfficientViT-Seg, chỉ khi cả hai đã được fine-tune/evaluate trên **cùng tập segmentation**.

> **Cấm so sánh trực tiếp** Accuracy của ViT classification với Dice/mIoU của EfficientViT segmentation và kết luận model nào tốt hơn. Đó là hai nhiệm vụ khác nhau.

### Định nghĩa bài toán — framework tổng quát, KHÔNG chứa tên model

**Classification:** ảnh X-quang → kiểm tra và tiền xử lý → trích xuất đặc trưng → bộ phân lớp → xác suất/nhãn dự đoán.

- Input: ảnh X-quang và metadata cần thiết.
- Ground truth: nhãn ảnh (FracAtlas: fracture / non-fracture; BTXRD: normal / benign / malignant).
- Output: vector score/probability theo lớp và nhãn dự đoán.
- Loss sau này: Cross-Entropy cho phân lớp loại trừ nhau; có thể cân nhắc trọng số lớp khi mất cân bằng.

**Segmentation:** ảnh X-quang → kiểm tra và tiền xử lý → encoder trích xuất đặc trưng → decoder không gian → mask theo pixel.

- Input: ảnh X-quang, annotation/mask tương ứng.
- Ground truth: pixel-level mask đã kiểm tra alignment.
- Output: mask dự đoán cùng hệ quy chiếu/kích thước mong muốn.
- Loss sau này: BCE/Dice hoặc loss thích hợp theo số lớp; chọn theo label semantics.

---

## 2. Chính xác đang nghiên cứu thuật toán nào?

| Hạng mục | Paper / cơ chế | Vai trò trong project | Triển khai ở Phase 0 |
|---|---|---|---|
| **ViT** | Dosovitskiy et al., *An Image Is Worth 16x16 Words*, ICLR 2021; patch tokens + global MSA | Mốc phân lớp chính | Adapter/interface + config, chưa train |
| **EfficientViT** | Han Cai et al., *Lightweight Multi-Scale Attention for High-Resolution Dense Prediction*, ICCV 2023; multi-scale linear attention | Đối tượng so sánh hiệu quả tính toán | Adapter/interface + config, chưa train |
| **DeiT** | Touvron et al., ICML 2021; training data-efficient + distillation | Related work / đối chứng tùy chọn | Link và ghi chú |
| **T2T-ViT, CaiT, Swin** | tokenization, deeper ViT, shifted windows | Lịch sử cải tiến kiến trúc | Link và ghi chú |
| **TransUNet** | Transformer encoder + U-Net-style decoder | Đối chứng segmentation ViT-based | Placeholder nhánh segmentation |
| **SegFormer, SegNeXt, TopFormer, SeaFormer** | Các hướng segmentation tối ưu khác | Related works / benchmark paper | Link và ghi chú |

**Phân biệt quan trọng:** *EfficientViT* của **Han Cai (ICCV 2023)** **khác** với *EViT — Expediting Vision Transformers* dựa trên pruning token. Không sử dụng lẫn hai kiến trúc chỉ vì cách viết tắt tương tự. Nếu về sau muốn nghiên cứu EViT pruning, hãy tạo track riêng.

### Ghi chú toán học để viết `docs/algorithms/` (chưa bắt Codex code)

- Standard self-attention: `softmax(QKᵀ / sqrt(d))V`, khi số token `N`, chiều head `d`, thành phần attention có `O(N²d)` phép tính và ma trận attention `N×N`.
- EfficientViT (Cai 2023): dùng feature mapping liên quan `ReLU(Q), ReLU(K)` và tính kết hợp để tránh tạo ma trận `N×N`, cùng aggregation đa tỉ lệ bằng depthwise/group convolution. Chi phí phần linear attention có thể viết theo `O(N d²)` nếu coi `d` cố định; **không** khẳng định toàn bộ network luôn có độ phức tạp O(N), vì projection, convolution và decoder còn chi phí riêng.
- Khi benchmark, phải **đo thực tế latency/VRAM**, không suy từ big-O rằng EfficientViT chắc chắn nhanh hơn trên mọi phần cứng.

---

## 3. Nguồn paper và mã nguồn chính thức

**Core (bắt buộc dẫn nguồn trong báo cáo và `source_registry.md`):**

| Chủ đề | Paper | Mã nguồn / tài liệu thực thi |
|---|---|---|
| Vision Transformer | https://arxiv.org/abs/2010.11929 | https://github.com/google-research/vision_transformer (repo gốc JAX/Flax) |
| ViT dùng trong PyTorch | ViT gốc ở trên | https://huggingface.co/docs/transformers/tasks/image_classification ; https://github.com/huggingface/pytorch-image-models (timm) |
| EfficientViT ICCV 2023 | https://openaccess.thecvf.com/content/ICCV2023/html/Cai_EfficientViT_Lightweight_Multi-Scale_Attention_for_High-Resolution_Dense_Prediction_ICCV_2023_paper.html | https://github.com/mit-han-lab/efficientvit |
| EfficientViT Classification | Cùng paper EfficientViT | https://github.com/mit-han-lab/efficientvit/tree/master/applications/efficientvit_cls |
| EfficientViT Segmentation | Cùng paper EfficientViT | https://github.com/mit-han-lab/efficientvit/tree/master/applications/efficientvit_seg |
| DeiT | https://proceedings.mlr.press/v139/touvron21a.html | https://github.com/facebookresearch/deit |
| Swin Transformer | https://arxiv.org/abs/2103.14030 | https://github.com/microsoft/Swin-Transformer |
| TransUNet | https://arxiv.org/abs/2102.04306 | https://github.com/Beckschen/TransUNet |
| SegFormer | https://arxiv.org/abs/2105.15203 | https://github.com/NVlabs/SegFormer |
| SegNeXt | https://arxiv.org/abs/2209.08575 | https://github.com/Visual-Attention-Network/SegNeXt |
| TopFormer | https://arxiv.org/abs/2204.05525 | https://github.com/hustvl/TopFormer |
| SeaFormer | https://openreview.net/forum?id=-qg8MQNrxZw | https://github.com/fudan-zvg/SeaFormer |

**Datasets (không tự tải trong Phase 0):**

| Dataset | Chức năng | Link |
|---|---|---|
| **FracAtlas** | X-quang xương; nhãn fracture và annotation phục vụ localization/segmentation | https://github.com/XLR8-07/FracAtlas ; https://www.nature.com/articles/s41597-023-02432-4 |
| **BTXRD** | X-quang u xương; normal, benign, malignant và dữ liệu annotation | https://github.com/SHUNHANYAO/BTXRD ; https://www.nature.com/articles/s41597-024-04311-y |

**Yêu cầu với Codex:** ghi link/DOI/loại task/license/commit hoặc version **nếu đã xác minh** trong `docs/references/source_registry.md`; khi chưa xác minh, ghi `TODO_VERIFY` thay vì điền thông tin bịa. Repo EfficientViT chính thức tuy có các ứng dụng classification/segmentation, nhưng không mặc định checkpoint segmentation đã phù hợp với nhãn fracture của FracAtlas; muốn dùng để dự đoán vùng gãy phải có bước adaptation/fine-tune hợp lệ.

---

## 4. Công nghệ mục tiêu và các nguyên tắc setup

- **Python**: ưu tiên 3.10 hoặc 3.11 để tương thích rộng. Kiểm tra các package trước khi khóa phiên bản cụ thể.
- **Framework**: PyTorch; `torchvision`; `timm` và/hoặc Hugging Face Transformers khi cần. Repo gốc ViT là JAX/Flax chỉ để đối chiếu; không bắt buộc chuyển cả project sang JAX.
- **Dataset/Image**: Pillow, NumPy; Pandas; YAML config; scikit-learn dùng cho metric/split trong giai đoạn sau.
- **Dev tooling**: `pytest`, `ruff`, `pyproject.toml`; logging bằng `logging` chuẩn; có thể thêm `pre-commit` nếu dễ duy trì.
- **GPU**: tùy chọn. Phase 0 phải hoạt động trên CPU và trên máy chưa cài CUDA.
- **Config-driven**: không hardcode đường dẫn Windows, đường dẫn Colab, batch size hoặc model name trong từng file Python.
- **Reproducibility**: chuẩn bị chỗ cho random seed, split manifests, experiment config snapshot, environment versions, checkpoint metadata.

**Không cài đồng thời nhiều hệ dependency nặng (CUDA/JAX/TensorRT/MMCV) chỉ vì chúng tồn tại trong repo tham khảo.** Phase 0 chỉ cần dependency để project import và CLI `doctor` chạy, còn gói train là optional/extra. Tạo hướng dẫn cài PyTorch theo CPU/CUDA phù hợp với máy thay vì đóng đinh một CUDA wheel cho tất cả.

---

## 5. Cây thư mục Codex PHẢI tạo

> Dùng `src/` layout của một Python package. `.gitkeep` cho thư mục trống. Các file `*.py` chưa phải implement nghiệp vụ: có docstring/TODO và contract. Có thể tinh giản một vài file cấu hình nếu có lý do, nhưng **không bỏ các nhóm chức năng**.

```text
xla-vit-efficientvit/
├── README.md                         # Tài liệu sử dụng thực tế của repo (Codex tự viết sau scaffold)
├── AGENTS.md                         # Quy ước làm việc & giới hạn task cho Codex
├── pyproject.toml                    # package, optional deps, pytest, ruff config
├── .gitignore
├── .env.example                      # chỉ placeholder, KHÔNG có secret
├── .python-version                   # nếu chọn dùng
├── requirements/                     # có thể dùng pyproject extras thay thế
│   ├── README.md
│   └── README-cuda.md
│
├── configs/
│   ├── datasets/
│   │   ├── fracatlas.yaml            # metadata + field mapping placeholder
│   │   └── btxrd.yaml
│   ├── models/
│   │   ├── vit_b16.yaml
│   │   ├── efficientvit_b0.yaml
│   │   └── efficientvit_seg.yaml
│   └── experiments/
│       ├── fracatlas_classification.yaml
│       └── fracatlas_segmentation.yaml
│
├── src/
│   └── xla/
│       ├── __init__.py
│       ├── __main__.py               # python -m xla ...
│       ├── cli.py                    # CLI skeleton; doctor thực thi được
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py             # load/validate config nhẹ
│       │   ├── paths.py              # pathlib; paths tương đối repo/config
│       │   ├── logging_utils.py
│       │   ├── seed.py
│       │   └── types.py              # shared dataclasses / type hints
│       ├── data/
│       │   ├── __init__.py
│       │   ├── datasets.py           # Dataset interface
│       │   ├── fracatlas.py          # adapter (TODO data parsing)
│       │   ├── btxrd.py              # adapter (TODO data parsing)
│       │   ├── inspect.py            # planned statistics
│       │   ├── preprocess.py         # planned image/label transforms
│       │   ├── splitting.py          # planned leakage-aware split
│       │   └── masks.py              # planned annotation to mask
│       ├── models/
│       │   ├── __init__.py
│       │   ├── registry.py           # model name/task -> adapter
│       │   ├── vit.py                # ViT classification adapter TODO
│       │   ├── efficientvit.py       # EfficientViT classification adapter TODO
│       │   ├── transunet.py          # OPTIONAL segmentation adapter TODO
│       │   └── segmentation.py       # EfficientViT-Seg adapter TODO
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── trainer.py            # planned training contract
│       │   ├── evaluator.py          # planned evaluation contract
│       │   ├── inferencer.py         # planned inference contract
│       │   ├── losses.py             # future loss selection
│       │   └── checkpoints.py        # future checkpoint metadata contract
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── classification.py     # Acc/Precision/Recall/F1/AUROC future
│       │   ├── segmentation.py       # Dice/IoU future
│       │   └── efficiency.py         # params/MACs/latency/memory future
│       └── experiments/
│           ├── __init__.py
│           ├── attention_scaling.py  # planned O(N^2) vs linear demonstration
│           └── benchmark.py          # planned fixed-protocol performance runs
│
├── scripts/
│   ├── README.md                     # lệnh chạy / trạng thái Phase 0–4
│   └── prepare_data_placeholder.py   # OPTIONAL, không làm download ngầm
│
├── tests/
│   ├── conftest.py
│   ├── test_imports.py
│   ├── test_cli.py
│   ├── test_config.py
│   └── fixtures/
│       └── README.md                 # cách tạo synthetic tiny test fixtures
│
├── docs/
│   ├── project_scope.md
│   ├── architecture.md               # pipeline chung, không cột cứng vào ViT
│   ├── data_protocol.md              # nhãn, split, leakage, masks
│   ├── experiment_protocol.md        # đánh giá công bằng, chưa có số liệu
│   ├── roadmap.md
│   ├── decisions.md                   # assumptions/ADRs/open questions
│   ├── algorithms/
│   │   ├── vit.md
│   │   └── efficientvit.md
│   └── references/
│       ├── papers.md
│       └── source_registry.md
│
├── data/
│   ├── README.md                     # đặt dataset đúng chỗ + nguồn/giấy phép
│   ├── raw/                          # KHÔNG COMMIT ảnh/nhãn gốc
│   │   ├── fracatlas/.gitkeep
│   │   └── btxrd/.gitkeep
│   ├── interim/.gitkeep
│   ├── processed/.gitkeep
│   └── splits/.gitkeep
│
├── third_party/
│   └── README.md                     # link/license/commit, chưa vendor code
│
├── outputs/
│   ├── checkpoints/.gitkeep
│   ├── logs/.gitkeep
│   ├── figures/.gitkeep
│   └── metrics/.gitkeep
│
├── reports/
│   ├── README.md                     # báo cáo theo 5 chương thầy yêu cầu
│   └── figures/.gitkeep
│
└── .github/
    └── workflows/
        └── ci.yml                    # optional: lint/test CPU only
```

**Lưu ý:** Khi tạo repo thật, `README.md` mới ở root phải hướng dẫn sử dụng và liên kết tới blueprint này (có thể đặt tài liệu blueprint vào `docs/CODEX_SETUP_SPEC.md`). Đừng xóa blueprint sau khi hoàn thành scaffold.

---

## 6. Hợp đồng trách nhiệm module — để các thành viên code sau

| Module | Đầu vào tương lai | Đầu ra tương lai | Trách nhiệm |
|---|---|---|---|
| `data/fracatlas.py` | dataset path + metadata | image, label, mask?, sample_id | parsing theo format thật, kiểm tra nhãn |
| `data/btxrd.py` | dataset path + metadata | image, label, mask?, sample_id | giống trên, phân biệt 3 lớp |
| `data/splitting.py` | metadata, seed, split strategy | train/val/test manifests | tránh leakage; tránh duplicate/patient-overlap nếu metadata cho phép |
| `data/preprocess.py` | image + train/eval mode | model-ready tensor | resize, normalization, train-only augmentation phù hợp y khoa |
| `data/masks.py` | annotation, image dimensions | aligned pixel mask | map chính xác tọa độ/nhãn, kiểm tra mask trống |
| `models/registry.py` | task/model id/config | model adapter | thống nhất interface `build_model()` |
| `models/vit.py` | config (num_classes, pretrained...) | ViT classifier | có thể dùng `timm`/Transformers, không yêu cầu JAX |
| `models/efficientvit.py` | config | EfficientViT classifier | ưu tiên implementation đúng nhóm Cai 2023 |
| `models/transunet.py` | segmentation config | segmentation model | nhánh mở rộng; không coi là ViT classifier |
| `engine/trainer.py` | model, loader, loss, optimizer | checkpoint + history | training loop; chưa implement Phase 0 |
| `engine/evaluator.py` | predictions + labels | metrics/figures | tuyệt đối không dùng test cho model selection |
| `metrics/efficiency.py` | model + input spec + device | MACs, params, latency, memory | chuẩn benchmark nhất quán |
| `experiments/attention_scaling.py` | tokens N, head dim d | time/memory curves | giải thích complexity với dữ liệu synthetic có ghi rõ |

**Interface convention đề xuất cho các phase sau:**

- Dataset item chứa tối thiểu `image`, `target`, `sample_id`; segmentation có thêm `mask` và `original_size` khi cần.
- `ModelAdapter` trả về output khác nhau theo `task`: class logits `[B,C]` hoặc segmentation logits `[B,C,H,W]`. Không ép chúng vào cùng một metric.
- `RunResult`/artifact chứa config snapshot, seed, dataset manifest hash, model ID, environment và metrics. Test không được tự sinh metrics giả rồi gắn nhãn "real experiment".
- Dùng `pathlib.Path`, type hints, docstrings, các lỗi có ngữ cảnh. Không in thông tin máy cá nhân hoặc secret ra log.

---

## 7. Các file config Codex cần tạo ở Phase 0

Tạo YAML **hợp lệ về mặt cú pháp**, có phần comment chỉ rõ trường nào là `TODO_VERIFY` vì còn phụ thuộc cấu trúc dataset thật. Không bịa đường dẫn hay mean/std của tập X-quang.

**Các trường tối thiểu:**

- Dataset config: `name`, `root`, `task`, `class_names`, `metadata_file`/`annotations_dir` (để trống nếu chưa biết), `split.strategy`, `split.seed`, `input_size`, `normalization`.
- Model config: `name`, `task`, `source`, `pretrained`, `num_classes`, `input_channels`, `input_size`, `notes`.
- Experiment config: `experiment_name`, `dataset_config`, `model_config`, `seed`, `device` (`auto`), `output_dir`, `batch_size`, `num_workers`, `learning_rate`, `epochs`, `metrics` (chỉ để dành cho Phase 2+).

**Chưa được** ngầm biến annotation segmentation thành labels classification; phải kiểm tra taxonomy/định nghĩa gốc của từng dataset. Cấu hình label mapping có thể chỉ placeholder, nhưng phải được kiểm thử trước khi train thật.

---

## 8. Yêu cầu setup & CLI tối thiểu (Phase 0 cần CHẠY ĐƯỢC)

### 8.1 Tạo môi trường độc lập

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

**Linux / Colab local repo:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

- Đảm bảo `pyproject.toml` định nghĩa extra `dev`; lệnh cài này **phải hoạt động** sau khi Codex scaffold.
- Nếu cần Torch cho `dev`, nên làm rõ chi phí cài đặt; ưu tiên không ép install GPU-specific packages ở Phase 0.
- Hướng dẫn riêng cách cài PyTorch CPU/GPU phù hợp tại `requirements/README-cuda.md` (link tài liệu chính thức https://pytorch.org/get-started/locally/). Không mặc định máy nào cũng có NVIDIA GPU.

### 8.2 CLI base

Các lệnh **phải hoạt động ngay**:

```bash
python -m xla --help
python -m xla doctor
python -m xla config-check --config configs/experiments/fracatlas_classification.yaml
python -m pytest -q
python -m ruff check .
```

- `doctor`: in Python version, OS, project root, packages tối thiểu, nếu có Torch thì hiện version và CUDA available; ghi `not installed`/`not available` nếu thiếu, không crash. Exit code khác 0 khi thiếu dependency **bắt buộc của Phase 0**, còn GPU không phải yêu cầu.
- `config-check`: parse YAML, kiểm tra schema/đường dẫn tới **các file config** (không ép phải có dataset thật), báo trường `TODO_VERIFY` nếu có.
- `--help`: hiển thị mô tả và các subcommand; có thể dùng `argparse`, Typer hoặc Click, ưu tiên ít dependency.
- Các lệnh `train`, `evaluate`, `predict`, `benchmark`, `prepare-data` chỉ cần **stub với thông báo "Not implemented — planned in Phase ..."** hoặc không đăng ký nếu chưa hỗ trợ; **không** tạo kết quả giả.

### 8.3 Kiểm tra smoke test

- Có ít nhất 3 test: package import, CLI help/doctor, đọc config YAML mẫu.
- Fixtures synthetic cực nhỏ **chỉ dùng cho unit test**, đánh dấu rõ `synthetic`, không được đưa vào kết quả đồ án.
- `pytest` phải chạy được **không cần** raw dataset, pretrained weight, GPU hay Internet.
- `ruff` phải pass trên source do Codex vừa tạo.

---

## 9. Quy chuẩn dữ liệu & thực nghiệm để giữ sẵn chỗ trong thiết kế

### Classification (ưu tiên)

1. Bắt đầu với **FracAtlas**; dùng train/val/test manifest cố định, cùng split cho ViT và EfficientViT.
2. Kiểm tra cùng bệnh nhân/ảnh gần trùng có thể nằm ở nhiều split không; nếu dataset không có patient ID, **nêu hạn chế** thay vì tuyên bố patient-level split.
3. Nếu bị mất cân bằng lớp, báo class counts và sử dụng macro-F1/precision/recall cùng accuracy. Không chỉ báo accuracy.
4. Fine-tune từ pretrained khi phù hợp, nhưng mô tả rõ nguồn pretrained, input resolution và preprocessing mỗi model.
5. Để so sánh công bằng, cố định chính sách split, device, precision, warmup, số lần đo, protocol chọn checkpoint. So sánh cả **model-default** và **matched input resolution** nếu khác kích thước input; ghi rõ từng chế độ để không lẫn.

### Segmentation (chỉ roadmap)

1. Mask GT cần đúng orientation/scale/label; cùng split và cùng tiêu chí đo cho hai model segmentation.
2. Đánh giá Dice, IoU, precision/recall theo mask; xử lý rõ trường hợp ảnh âm tính / mask trống.
3. Không dùng pretrained trên ADE20K/Cityscapes rồi gọi dự đoán trực tiếp là kết quả fracture segmentation nếu chưa fine-tune đúng nhãn.

### Benchmark efficiency

- Đo Params, MACs (nêu rõ tool và phép đếm), peak GPU memory, latency/throughput.
- Ghi hardware/OS/framework/device, batch size, resolution, FP32/FP16, warmup, sync CUDA, thời gian preload/data transfer có tính hay không, số lần đo và median/p50/p95.
- Thí nghiệm Attention Scaling là **minh họa lý thuyết**; không dùng đồ thị synthetic để kết luận tốc độ của toàn ViT so với toàn EfficientViT.

Các nguyên tắc ở phần này là **hợp đồng cho giai đoạn sau**, Phase 0 chỉ tạo chỗ chứa config/tài liệu/test interface.

---

## 10. Roadmap — Codex KHÔNG làm lẫn giai đoạn

### Phase 0 — Scaffold & environment (**LÀM NGAY**)

- [ ] Kiểm tra repo hiện tại; không ghi đè code có sẵn.
- [ ] Tạo cây thư mục Mục 5; `.gitkeep` cho thư mục rỗng.
- [ ] Tạo `pyproject.toml`, `.gitignore`, `.env.example`, `AGENTS.md`.
- [ ] Tạo Python package `src/xla` với modules placeholder có docstring/TODO.
- [ ] Tạo `configs/` và YAML syntactically valid cho dataset/model/experiment.
- [ ] Làm CLI `--help`, `doctor`, `config-check` chạy thật.
- [ ] Tạo `tests/`; chạy được `pytest`, `ruff` offline.
- [ ] Viết `README.md` sử dụng repo; ghi rõ **base only / chưa có model training**.
- [ ] Viết `docs/architecture.md`, `docs/roadmap.md`, `docs/references/source_registry.md`, `docs/decisions.md`.
- [ ] Kiểm tra không commit dataset/weights/.env/outputs/venv.

### Phase 1 — Dataset pipeline (**CHƯA LÀM**)

- Inspect cấu trúc FracAtlas thực tế → label mapping → masks nếu cần → kiểm tra trùng lặp → manifests → DataLoader → visualization.
- Chưa biết dữ liệu thực tế thì không thể tự khẳng định data loader là đúng.

### Phase 2 — ViT & EfficientViT classification (**CHƯA LÀM**)

- Cài model adapters đúng nguồn; chạy forward, train, val, test, checkpoint, inference. Có baseline/seed/splits/logs rõ ràng.

### Phase 3 — Complexity & fair comparison (**CHƯA LÀM**)

- Attention scaling microbenchmark + model benchmark, FLOPs/MACs, latency/VRAM, metric tables/plots.

### Phase 4 — Segmentation / BTXRD extension (**CHƯA LÀM**)

- Segmentation 2 mô hình cùng dataset/GT; hoặc mở rộng classification BTXRD; so sánh kết quả riêng từng tác vụ.

---

## 11. Definition of Done — tiêu chí nghiệm thu Phase 0

Codex chỉ được báo "Phase 0 completed" nếu:

1. Cây thư mục tối thiểu và `pyproject.toml` tồn tại, project có thể cài editable từ root.
2. `python -m xla --help` trả exit code 0.
3. `python -m xla doctor` chạy không cần dataset, pretrained checkpoint hay GPU.
4. `python -m xla config-check --config configs/experiments/fracatlas_classification.yaml` đọc config, báo lỗi hợp lý nếu YAML hỏng và không fail vì data chưa tải.
5. `python -m pytest -q` pass; `python -m ruff check .` pass.
6. Không có tập dữ liệu thật, checkpoint pretrained, token/secret hay file output lớn bị commit nhầm.
7. `README.md` mới hướng dẫn setup trên Windows/Linux và chỉ rõ phần nào đã/chưa được implement.
8. `docs/references/source_registry.md` phân biệt **link paper**, **official source**, **framework adapter** và **chưa kiểm chứng**; không nhận code tham khảo là code do nhóm tự sáng tạo.
9. Kết thúc, Codex đưa báo cáo ngắn gồm `created/modified files`, kết quả command đã thực chạy, và 3 việc tiếp theo của Phase 1.

Nếu một điều kiện không thể hoàn tất do môi trường hạn chế: báo rõ **blocked** + lý do + hướng giải quyết, không đánh dấu pass giả.

---

## 12. Hướng dẫn gửi task cho Codex

Dán nguyên văn prompt ngắn sau vào Codex **sau khi đã đặt file này ở root repo** (hoặc đính kèm file này):

> Đọc kỹ `README_CODEX_SETUP_XLA.md` làm tài liệu đặc tả. Hãy **chỉ triển khai PHASE 0**: scaffold kiến trúc dự án ViT/EfficientViT, setup Python package, configs, docs, CLI doctor/config-check và test offline theo cây thư mục + Definition of Done trong file. Trước tiên inspect repo để không ghi đè code hiện có. Không train, không download dataset/weights, không tự implement toàn bộ mô hình hay báo metric giả. Sau khi tạo xong, chạy các lệnh kiểm tra thực tế, sửa lỗi và báo cáo những gì đã pass/blocked. Nếu cần quyết định ngoài đặc tả, ghi vào `docs/decisions.md`.

**Đầu ra mong muốn của task này:** một repo sạch, có cấu trúc hợp lý, cài đặt được, sẵn sàng để các thành viên implement Phase 1 và Phase 2 — **không phải** một hệ thống nghiên cứu đã chạy hoàn chỉnh.
