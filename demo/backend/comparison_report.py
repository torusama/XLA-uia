"""Build charts and a reproducible ZIP from a completed paired comparison."""
import csv
import io
import json
import re
import threading
from collections import defaultdict
from functools import lru_cache
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile

from .history import METRIC_KEYS

MODELS = ("vit", "efficientvit")
LABELS = ("ViT", "EfficientViT")
COLORS = ("#0f766e", "#ea7257")
CHARTS = {
    "accuracy.png": "Độ chính xác và dự đoán nhầm",
    "paired-dice.png": "Dice từng ảnh có u",
    "latency.png": "Phân bố độ trễ và thông lượng",
    "complexity.png": "Số tham số và GFLOPs",
    "resources.png": "RAM, CPU và VRAM",
}
REPORT_LOCK = threading.Lock()


def get_report(history, job_id):
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise LookupError("Lượt so sánh không tồn tại")
    # Matplotlib font/render state is shared by all HTTP threads.
    with REPORT_LOCK:
        return build_report(history, job_id)


@lru_cache(maxsize=3)
def build_report(history, job_id):
    with history.connect() as connection:
        job = connection.execute("SELECT * FROM batch_jobs WHERE id=?", (job_id,)).fetchone()
    if job is None:
        raise LookupError("Lượt so sánh không tồn tại")
    if not job["selection"] or job["state"] != "complete" or job["completed"] != job["total"]:
        raise ValueError("Cần hoàn tất lượt so sánh trên tập ảnh đã kiểm tra")
    selection = json.loads(job["selection"])
    data = history.query(limit=2000, batch_id=job_id)
    rows = data["rows"]
    pairs = defaultdict(list)
    by_model = defaultdict(list)
    for row in rows:
        pairs[row["sample_index"]].append(row)
        by_model[row["model"]].append(row)
    expected = {sample["index"]: sample for sample in selection["included"]}
    if (set(expected) != set(pairs) or len(expected) != job["total"] or len(rows) != 2 * job["total"]
            or any({row["model"] for row in pair} != set(MODELS)
                   or len({row["group_id"] for row in pair}) != 1
                   or any(row["sample_label"] != expected[index]["label"] for row in pair)
                   for index, pair in pairs.items())):
        raise ValueError("Số liệu không đủ hai mô hình cho từng ảnh trong danh sách")
    if sorted(summary["model"] for summary in data["summaries"]) != sorted(MODELS):
        raise ValueError("Lượt chạy có nhiều cấu hình đo cho cùng mô hình")
    summaries = {summary["model"]: summary for summary in data["summaries"]}
    for model, summary in summaries.items():
        tumor = [row for row in by_model[model] if row["sample_label"] == "tumor"]
        normal = [row for row in by_model[model] if row["sample_label"] == "normal"]
        summary["tumor_mean_dice"] = mean(row["dice"] for row in tumor) if tumor else None
        summary["tumor_mean_iou"] = mean(row["iou"] for row in tumor) if tumor else None
        summary["normal_false_positive_count"] = sum(row["fp"] > 0 for row in normal)
        summary["normal_count"] = len(normal)
        summary["normal_false_positive_rate"] = (summary["normal_false_positive_count"] / len(normal)
                                                  if normal else None)
    overview = {"job_id": job_id, "created_at": job["created_at"], "completed_at": job["updated_at"],
                "image_count": len(expected), "included_by_label": selection["included_by_label"],
                "charts": CHARTS, "summaries": summaries}
    files = render_charts(summaries, by_model)
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(f"charts/{name}", content)
        archive.writestr("summary.json", json.dumps(overview, indent=2, ensure_ascii=False))
        archive.writestr("selection.json", json.dumps(selection, indent=2, ensure_ascii=False))
        archive.writestr("per-image.csv", make_csv(
            rows, ["sample_index", "sample_name", "sample_label", "model", "created_at", *METRIC_KEYS]))
        fields = [field for field in summaries["vit"] if field != "config"]
        archive.writestr("summary.csv", make_csv(summaries.values(), fields))
        archive.writestr("method.txt",
                         "Image selection: decode, format, checksum, annotation and mask validation, "
                         "then retain one copy per identical image hash. No model scores enter selection.\n"
                         "Both models predict the same 224×224 images in fixed ViT-first order "
                         "with one timed forward pass per model per image. No flip ensemble.\n"
                         "Micro Dice/IoU pools foreground pixels from all included images. "
                         "Tumor means average only positive images. A normal image is flagged when "
                         "it has at least one predicted foreground pixel.\n"
                         "Latency p95 uses nearest rank. Throughput is images / total forward time. "
                         "It excludes loading, rendering, and extra FLOP profiling.\n"
                         "Boxplots: median, quartiles and Tukey 1.5×IQR whiskers. Outliers appear "
                         "in the CSV. RSS covers the full process. CPU is measured in equivalent "
                         "cores and VRAM is tensor memory above the baseline. Blank = unavailable.\n")
    files["report.zip"] = buffer.getvalue()
    return overview, files


def make_csv(rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def render_charts(summaries, rows):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    files = {}

    def bars(axis, key, title, unit, upper=None):
        values = [summaries[model].get(key) for model in MODELS]
        for index, value in enumerate(values):
            if value is not None:
                bar = axis.bar(index, value, color=COLORS[index], width=0.55)
                axis.bar_label(bar, labels=[f"{value:.3g}"], padding=4)
            else:
                axis.text(index, 0.05, "Thiếu số đo", ha="center", transform=axis.get_xaxis_transform())
        axis.set_xticks(range(len(LABELS)), LABELS)
        axis.set(title=title, ylabel=unit)
        axis.set_ylim(0, upper or max((value or 0) for value in values) * 1.25 or 1)

    def boxes(axis, key, title, unit):
        available = False
        for index, model in enumerate(MODELS):
            values = [row[key] for row in rows[model] if row[key] is not None]
            if values:
                available = True
                axis.boxplot(values, positions=[index], widths=0.46, patch_artist=True,
                             boxprops={"facecolor": COLORS[index], "alpha": 0.65},
                             medianprops={"color": "#172c35", "linewidth": 2}, showfliers=False)
            else:
                axis.text(index, 0.4, "—", ha="center", transform=axis.get_xaxis_transform())
        axis.set_xticks(range(len(LABELS)), LABELS)
        axis.set(xlim=(-0.55, 1.55), title=title, ylabel=unit)
        axis.set_ylim(bottom=0)
        if not available:
            axis.text(0.5, 0.6, "Không có số đo", ha="center", transform=axis.transAxes)

    def paired_dice(axis):
        scores = [{row["sample_index"]: row["dice"] for row in rows[model]
                   if row["sample_label"] == "tumor"} for model in MODELS]
        indices = sorted(scores[0])
        axis.plot([0, 1], [0, 1], "--", color="#7a909c", label="Điểm bằng nhau")
        axis.scatter([scores[0][i] for i in indices], [scores[1][i] for i in indices],
                     color=COLORS[0], alpha=0.5, edgecolors="none", s=17)
        axis.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="Dice ViT",
                 ylabel="Dice EfficientViT", aspect="equal")
        axis.legend(loc="upper left", frameon=False)

    def chart(name, panels):
        figure = Figure(figsize=(max(4.6, 3.3 * len(panels)), 3.5),
                        layout="constrained", facecolor="white")
        figure.suptitle(CHARTS[name], fontsize=13, fontweight="bold", color="#172c35")
        for axis, (draw, *args) in zip(figure.subplots(1, len(panels), squeeze=False)[0], panels):
            draw(axis, *args)
            axis.set_axisbelow(True)
            axis.grid(axis="y", color="#e0e8eb", linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
        canvas = FigureCanvasAgg(figure)
        buffer = io.BytesIO()
        canvas.print_png(buffer)
        files[name] = buffer.getvalue()

    chart("accuracy.png", [(bars, "micro_dice", "Dice gộp", "Điểm", 1.12),
                           (bars, "micro_iou", "IoU gộp", "Điểm", 1.12),
                           (bars, "normal_false_positive_rate", "Ảnh thường tô nhầm", "Tỷ lệ", 1.12)])
    chart("paired-dice.png", [(paired_dice,)])
    chart("latency.png", [(boxes, "latency_ms", "Độ trễ mỗi ảnh", "ms"),
                          (bars, "p95_ms", "Độ trễ p95", "ms"),
                          (bars, "throughput_ips", "Thông lượng forward", "ảnh/s")])
    chart("complexity.png", [(bars, "mean_params_m", "Số tham số", "triệu"),
                             (bars, "mean_gflops", "Phép tính mỗi lượt", "GFLOPs")])
    chart("resources.png", [(boxes, "rss_delta_mb", "RAM tăng thêm", "MiB"),
                            (boxes, "cpu_cores", "CPU", "lõi tương đương"),
                            (boxes, "gpu_delta_mb", "VRAM tăng thêm", "MiB")])
    return files
