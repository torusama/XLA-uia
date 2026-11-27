"""Serve BTXRD samples, ground truth, and optional model inference locally."""
import argparse
import csv
import io
import json
import os
import sqlite3
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .batch import BatchRunner
from .dataset import ROOT, TestDataset, png_bytes
from .history import METRIC_KEYS, HistoryStore
from .models import ModelRegistry


class DemoHandler(SimpleHTTPRequestHandler):
    dataset = None
    dataset_error = None
    registry = None
    history = None
    batch = None

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_bytes(self, content, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload, status=200):
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8", status)

    def do_GET(self):
        parsed = urlsplit(self.path)
        if not parsed.path.startswith("/api/"):
            return super().do_GET()
        if parsed.path == "/api/models":
            return self.send_json({"models": self.registry.statuses()})
        if parsed.path == "/api/batch":
            if self.batch is None:
                return self.send_json({"error": "Batch service unavailable"}, 503)
            return self.send_json({"jobs": self.batch.jobs(),
                                   "ready": all(self.registry.statuses()[name]["ready"]
                                                for name in ("vit", "efficientvit")),
                                   "total_samples": len(self.dataset.rows) if self.dataset else 0})
        if parsed.path in ("/api/history", "/api/history.csv"):
            if self.history is None:
                return self.send_json({"error": "History storage unavailable"}, 503)
            filters = parse_qs(parsed.query)
            try:
                data = self.history.query(limit=int(filters.get("limit", ["200"])[0]),
                                          session=filters.get("session", ["all"])[0],
                                          mode=filters.get("mode", ["compare"])[0],
                                          hours=int(filters.get("hours", ["0"])[0]),
                                          batch_id=filters.get("batch_id", [""])[0])
            except ValueError:
                return self.send_json({"error": "Invalid history filter"}, 400)
            if parsed.path.endswith(".csv"):
                stream = io.StringIO(newline="")
                fields = ["id", "session", "group_id", "batch_id", "mode", "created_at", "model", "config_id",
                          "sample_index", "sample_name", "sample_label", *METRIC_KEYS, "config"]
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for row in data["rows"]:
                    writer.writerow(dict(row, config=json.dumps(row["config"])))
                return self.send_bytes(stream.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8")
            return self.send_json(data)
        if self.dataset is None:
            return self.send_json({"error": self.dataset_error}, 503)
        try:
            if parsed.path == "/api/samples":
                samples = []
                for index in range(len(self.dataset.rows)):
                    sample = self.dataset.describe(index)
                    sample.update(image_url=f"/api/image?index={index}",
                                  ground_truth_url=f"/api/ground-truth?index={index}")
                    samples.append(sample)
                return self.send_json({"samples": samples})
            args = parse_qs(parsed.query)
            index = int(args.get("index", [""])[0])
            image_bytes, image, mask = self.dataset.read(index)
            if parsed.path == "/api/image":
                return self.send_bytes(image_bytes, "image/jpeg")
            if parsed.path == "/api/ground-truth":
                return self.send_bytes(png_bytes(mask), "image/png")
            if parsed.path == "/api/predict":
                model = args.get("model", [""])[0]
                if model not in ("vit", "efficientvit"):
                    return self.send_json({"error": "Unknown model"}, 400)
                result = self.registry.infer(model, image, mask)
                if self.history is not None:
                    self.history.record(self.dataset.describe(index), "single", {model: result})
                return self.send_json({"index": index, "model": model, "result": result})
            if parsed.path == "/api/compare":
                results = {name: self.registry.infer(name, image, mask)
                           for name in ("vit", "efficientvit")}
                if self.history is not None:
                    self.history.record(self.dataset.describe(index), "compare", results)
                return self.send_json({"index": index, "models": results})
            return self.send_json({"error": "Not found"}, 404)
        except (IndexError, ValueError):
            return self.send_json({"error": "Invalid sample index"}, 400)
        except (OSError, RuntimeError):
            return self.send_json({"error": "Could not process sample"}, 500)
        except sqlite3.Error:
            return self.send_json({"error": "Could not persist inference history"}, 503)

    def do_POST(self):
        path = urlsplit(self.path).path
        origin = self.headers.get("Origin")
        if (self.headers.get("X-Requested-With") != "BoneScope" or
                (origin and urlsplit(origin).netloc != self.headers.get("Host"))):
            return self.send_json({"error": "Same-origin application request required"}, 403)
        if path not in ("/api/batch/start", "/api/batch/stop"):
            return self.send_json({"error": "Not found"}, 404)
        if self.batch is None:
            return self.send_json({"error": "Batch service unavailable"}, 503)
        try:
            if path.endswith("start"):
                return self.send_json({"job_id": self.batch.start()}, 202)
            self.batch.stop()
            return self.send_json({"jobs": self.batch.jobs()})
        except ValueError as exc:
            return self.send_json({"error": str(exc)}, 409)
        except sqlite3.Error:
            return self.send_json({"error": "Could not save batch job"}, 503)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit-checkpoint", type=Path,
                        default=os.environ.get("XLA_VIT_CHECKPOINT"))
    default_efficientvit = ROOT / "outputs" / "checkpoints" / "best_evit_seg_btxrd.pth"
    parser.add_argument("--efficientvit-checkpoint", type=Path,
                        default=os.environ.get("XLA_EFFICIENTVIT_CHECKPOINT", default_efficientvit))
    parser.add_argument("--vit-threshold", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--history-db", type=Path, default=ROOT / "outputs/demo_history.sqlite3")
    args = parser.parse_args()
    try:
        DemoHandler.dataset = TestDataset()
    except (OSError, ValueError, KeyError) as exc:
        DemoHandler.dataset_error = str(exc)
        print(f"BTXRD unavailable: {exc}", flush=True)
    if not 0 < args.vit_threshold < 1:
        parser.error("vit-threshold must be between 0 and 1")
    DemoHandler.registry = ModelRegistry(args.vit_checkpoint, args.efficientvit_checkpoint,
                                         vit_threshold=args.vit_threshold)
    DemoHandler.history = HistoryStore(args.history_db)
    DemoHandler.batch = BatchRunner(DemoHandler.dataset, DemoHandler.registry, DemoHandler.history)
    handler = partial(DemoHandler, directory=str(ROOT / "demo/frontend"))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Demo available at http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
