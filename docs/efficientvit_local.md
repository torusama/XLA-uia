# EfficientViT cho demo cục bộ

- Nguồn: [mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit), commit `de7d7733cc0329f391b33f1f459271562ec27bd5`, giấy phép Apache-2.0.
- Môi trường demo đã cài package từ commit trên với `--no-deps`; các thư viện PyTorch, torchvision, timm, NumPy và Pillow do môi trường dự án cung cấp. Package upstream khai báo version `0.0.0`.
- Bộ nạp chỉ sử dụng phần segmentation B1. Trên môi trường không có Triton, lớp Triton RMSNorm không khả dụng và sẽ báo lỗi nếu bị gọi; adapter không sửa file trong thư viện đã cài. Checkpoint B1 đang dùng batch normalization và không cần lớp đó.
- Checkpoint `outputs/kaggle_checks/efficientvit/best_evit_seg_btxrd.pth` tải từ job [EfficientViT-Seg](https://www.kaggle.com/code/qunthy/efficientvit-seg). SHA-256 của file này khác checkpoint `outputs/checkpoints/best_evit_seg_btxrd.pth` theo Git; không gán kết quả của job cho file còn lại khi chưa đối chiếu.

Để cài lại sau khi cài các dependency ViT trong README gốc:

```powershell
python -m pip install --no-deps "efficientvit @ git+https://github.com/mit-han-lab/efficientvit.git@de7d7733cc0329f391b33f1f459271562ec27bd5"
```
