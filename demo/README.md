# Demo ViT và EfficientViT

## Cấu trúc

```text
demo/
  frontend/              Giao diện HTML, CSS và JavaScript
  backend/dataset.py     Đọc ảnh test và tạo mask ground truth từ polygon
  backend/models.py      Nạp checkpoint ViT, chạy model và đo chỉ số
  backend/batch.py       Chạy so sánh toàn bộ tập ảnh khi người dùng bấm nút
  backend/history.py     Lưu lịch sử, tiến độ và thống kê trong SQLite
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

Ngưỡng foreground có thể thay bằng `--vit-threshold`. Với checkpoint Segmenter epoch 87 của job 100 epoch, phép khảo sát trên validation chọn ngưỡng 0.8 trong lưới 0.10–0.95; đây là tham số đã chọn trên validation, không áp dụng mặc định cho checkpoint khác. Ví dụ, sau khi tải checkpoint tương ứng về `outputs/segmenter/best.pt`:

```powershell
python -m demo.backend.server --vit-checkpoint outputs/segmenter/best.pt --vit-threshold 0.8
```

Ngưỡng 0.8 làm giảm vùng tô dư và đồng thời bỏ sót nhiều hơn trên validation. Không tinh chỉnh ngưỡng dựa trên từng ảnh test trong demo.

EfficientViT-Seg-B1 đã có adapter. Xem [nguồn, phiên bản và cách cài](../docs/efficientvit_local.md). Ví dụ chạy cả hai checkpoint đã tải:

```powershell
python -m demo.backend.server --vit-checkpoint outputs/segmenter_weight50/segmenter_weight50_trial/best.pt --efficientvit-checkpoint outputs/kaggle_checks/efficientvit/best_evit_seg_btxrd.pth --vit-threshold 0.5
```

Checkpoint ViT trên là thí nghiệm weight 50; không dùng ngưỡng 0.8 đã chọn riêng cho checkpoint epoch 87 của thí nghiệm khác. Cần dùng cùng split, ground truth và giao thức đo khi so sánh hai model.

## Cách đọc chỉ số

Dice và IoU trong giao diện được tính **trên ảnh đang xem**, không phải điểm toàn bộ test. Khi cả mask dự đoán và ground truth đều rỗng, hai chỉ số được quy ước bằng 1. Thời gian là một lượt forward, đo bằng đồng hồ thực và đồng bộ CUDA khi dùng GPU; không bao gồm đọc ảnh/tiền xử lý, phụ thuộc thiết bị và có thể chậm ở lần đầu. Số tham số được đếm từ model đã nạp.

GFLOPs được đếm bằng `torch.utils.flop_counter.FlopCounterMode` cho một forward với batch 1, ảnh 224×224; một phép nhân cộng tính là hai FLOPs. Bộ đếm bao phủ các toán tử được PyTorch hỗ trợ như convolution, nhân ma trận và attention, không phải phép đếm đầy đủ mọi toán tử chuẩn hóa/nội suy/elementwise. Kết quả được cache theo model, shape, dtype và thiết bị; lượt profiling không nằm trong thời gian suy luận hiển thị. Nếu không đếm được, API trả `gflops_error` thay vì tạo số liệu. Muốn báo cáo kết quả toàn bộ test hoặc benchmark ổn định, cần chạy đánh giá riêng với cùng điều kiện cho hai model.

## API

- `GET /api/samples`: danh sách ảnh, nhãn và URL ảnh/ground truth.
- `GET /api/models`: trạng thái từng model.
- `GET /api/image?index=N` và `GET /api/ground-truth?index=N`: ảnh JPEG và mask PNG.
- `GET /api/predict?index=N&model=vit`: kết quả và chỉ số thực trên một ảnh.
- `GET /api/compare?index=N`: chạy các model sẵn sàng lần lượt trên cùng ảnh; model chưa có sẽ báo chưa sẵn sàng.
- `GET /api/history?session=all&mode=compare&limit=200&hours=0`: lịch sử và thống kê từ SQLite.
- `GET /api/history.csv` với cùng các query parameter: xuất CSV.
- `GET /api/batch`: trạng thái model, số ảnh và các tác vụ gần nhất; không khởi chạy tác vụ.
- `POST /api/batch/start` và `POST /api/batch/stop`: bắt đầu hoặc yêu cầu dừng; frontend gửi header `X-Requested-With: BoneScope` từ cùng origin.

## Xem lớn biểu đồ và so sánh toàn bộ tập ảnh

Mỗi panel có nút **Xem lớn** để mở biểu đồ trong hộp thoại rộng. Có thể xem tất cả số đo đang lọc hoặc 20/50 cặp gần nhất và mở bảng số đo chi tiết. Rê chuột hoặc đưa tiêu điểm bàn phím lên một điểm sẽ hiện thẻ nhỏ cạnh điểm đó, gồm model, giá trị và đơn vị, tên ảnh, thời điểm, lượt so sánh và thiết bị. Bấm/chạm điểm hoặc nhấn Enter/Space để ghim thẻ; chọn điểm khác để đổi thông tin. Thẻ đã ghim được giữ khi nhận số đo mới nếu điểm vẫn còn trong bộ lọc. Bấm Đóng trên thẻ, bấm ngoài hoặc nhấn Escape để đóng thẻ; nhấn Escape tiếp để đóng cửa sổ biểu đồ lớn. Biểu đồ lớn tiếp tục cập nhật khi có kết quả mới. Đóng cửa sổ bằng nút Đóng hoặc bấm bên ngoài.

Trong tab **So sánh**, bấm **So sánh toàn bộ 561 ảnh** khi muốn chạy cả hai model qua toàn bộ tập test đang có trên web. Mở trang, tải lại trang hoặc khởi động server không tự khởi chạy tác vụ này. Nút chỉ sẵn sàng khi cả hai checkpoint được nạp thành công. Server chỉ cho một tác vụ hàng loạt chạy tại một thời điểm.

Sau khi bấm, backend lần lượt chạy ViT rồi EfficientViT trên cùng ảnh, lưu cặp kết quả và tăng tiến độ sau mỗi ảnh hoàn tất. Frontend tự chọn bộ lọc tác vụ đó, hiển thị tối đa 2.000 dòng và cập nhật các biểu đồ. Với tập hiện tại, một lượt đầy đủ có 561 cặp, tức 1.122 dòng số đo. Nút **Dừng sau ảnh này** giữ lại các cặp đã hoàn tất. Có thể đóng hoặc tải lại trang trong lúc chạy; backend tiếp tục khi tiến trình server còn hoạt động. Khởi động lại server sẽ đánh dấu lượt dang dở là bị ngắt, giữ số đo đã lưu và không tự chạy lại.

Tiến độ và ID tác vụ được lưu trong cùng SQLite với lịch sử. Bộ lọc **Lượt chạy hàng loạt** và query `batch_id` tách kết quả mỗi lượt; không trộn với các lần chuyển ảnh thủ công. Bảng có Dice/IoU trung bình theo ảnh và Dice/IoU pixel gộp từ TP/FP/FN; chỉ số pixel chỉ tính các dòng có đủ bộ đếm, nên lịch sử cũ có thể chưa có. Một lượt bị dừng hoặc lỗi chỉ thống kê phần đã chạy; chỉ coi là đủ tập test khi trạng thái đã hoàn tất và đủ 561 ảnh.

## Lịch sử và tài nguyên

Cài `python -m pip install -e ".[vit,demo]"` để có dependency đo tài nguyên. `psutil` 7.2.2 (BSD-3-Clause, [nguồn](https://github.com/giampaolo/psutil)) được dùng để lấy RSS tiến trình. Thư viện được gọi qua dependency, không vendor code.

Server mặc định lưu lịch sử tại `outputs/demo_history.sqlite3` (Git bỏ qua), có thể đổi bằng `--history-db`. Không xóa dữ liệu khi server khởi động lại. Mỗi lần gọi predict thành công lưu một dòng; compare lưu hai dòng có chung group ID nếu cả hai model sẵn sàng. Một mẫu được chạy lại được tính là một lượt mới. Không lưu ảnh, mask base64 hoặc trọng số vào DB; chỉ lưu chỉ số, mẫu, nhãn, thời điểm UTC, session và cấu hình đo. Các lượt phát sinh trước khi thêm tính năng không được phục dựng thành số liệu giả. Có thể sao lưu DB bằng SQLite backup; khi sao chép thủ công hãy dừng server để WAL được đóng.

Trong tab So sánh, bên dưới bảng chỉ số ảnh đang xem có 10 panel riêng: thời gian, throughput suy luận thuần, Dice, IoU, RAM tiến trình, RAM tăng thêm, CPU, VRAM tăng thêm, tham số và GFLOPs. Mỗi panel có đơn vị, số của ViT và EfficientViT ở lượt gần nhất, biểu đồ và chú thích. Các độ đo biến thiên dùng đường, còn tham số và GFLOPs dùng cột vì là thuộc tính cố định của cấu hình. Có thể chọn trục theo lượt chạy hoặc thời gian; điểm trên đường có thông tin ảnh và thời điểm khi di chuột. Panel thiếu số đo vẫn hiển thị trạng thái chờ hoặc thông báo CPU chưa có VRAM.

Sau mỗi lần đổi ảnh và chạy xong hai model, frontend tự lấy dữ liệu mới từ backend và cập nhật biểu đồ; không cần chuyển tab hoặc bấm làm mới. Bảng tổng hợp hiển thị trung bình, trung vị, p95 kiểu nearest-rank và throughput `N / tổng thời gian forward`. Giá trị được tính trên tối đa 2.000 dòng so sánh gần nhất phù hợp bộ lọc, không phải toàn bộ DB khi vượt giới hạn. Có lọc phiên, khoảng thời gian và xuất CSV. Tổng hợp tách theo model, checkpoint SHA-256, ngưỡng, dtype, input, thiết bị, phần cứng, PyTorch và phiên bản phương pháp đo. Hai phiên có cùng cấu hình có thể được gộp nếu chọn tất cả phiên. Trong mỗi panel, hai số lớn và biểu đồ cột tham số/GFLOPs dùng cấu hình mới nhất của từng model trong bộ lọc; bảng tổng hợp vẫn tách mọi cấu hình.

Mỗi lần chuyển ảnh trong tab So sánh chạy hai model tuần tự trên cùng ảnh và thêm một điểm cho mỗi model vào chuỗi biểu đồ. Nếu đổi ảnh nhanh, lượt đã được backend tiếp nhận vẫn có thể hoàn tất và được lưu; frontend chỉ hiển thị mask của ảnh đang chọn. Lịch sử này phục vụ demo, không thay thế đánh giá test cố định. Lượt đầu có thể chịu ảnh hưởng warm-up; thứ tự ViT trước EfficientViT có thể gây thiên lệch nhiệt độ/cache. Không có số liệu điện năng hoặc GPU utilization.

Giới hạn phép đo tài nguyên:

- RAM: RSS tiến trình server, bao gồm cả hai model, được lấy mẫu mỗi 10 ms; có thể bỏ lỡ đỉnh ngắn. RAM tăng thêm là phần vượt mức RSS trước forward, không phải bộ nhớ riêng của model và không phải chỉ số rò rỉ bộ nhớ.
- CPU: CPU time toàn tiến trình chia wall time suy luận, biểu diễn số lõi CPU tương đương; không phải phần trăm CPU toàn máy. Hoạt động khác của server có thể ảnh hưởng phép đo.
- GPU: peak allocated tensor và phần tăng so với trước forward của PyTorch CUDA. Không bao gồm mọi VRAM driver hoặc phần mềm khác. Máy chạy CPU trả null, giao diện báo chưa có số đo.
- Profiler GFLOPs và tiền xử lý nằm ngoài thời gian forward. Resource sampling có overhead nhỏ. Tham số và GFLOPs cố định khi cấu hình model và kích thước input không đổi, nên được trình bày bằng biểu đồ cột.

Lịch sử ghi số đo bắt đầu từ phiên có tính năng này; khi thay checkpoint, khởi động lại backend bằng checkpoint tương ứng để metadata và model khớp nhau.

Bản web riêng tư được host tĩnh và có thể xem ảnh cùng ground truth. Model Python, checkpoint và GPU nằm trên máy chạy backend cục bộ; bản web tĩnh không tự chạy được model nếu chưa triển khai backend tương ứng. Không đưa checkpoint hoặc ảnh y tế riêng tư lên web tĩnh.

Để tạo lại bản web tĩnh từ dataset cục bộ, chạy `python -m demo.build_site`. Lệnh yêu cầu `demo/dist` chưa tồn tại và xuất 561 ảnh/mask vào đó; thư mục build được Git bỏ qua.
