"""Serve BTXRD samples, ground truth, and optional model inference locally."""
import argparse
import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .dataset import ROOT, TestDataset, png_bytes
from .models import ModelRegistry


class DemoHandler(SimpleHTTPRequestHandler):
    dataset = None
    dataset_error = None
    registry = None

    def send_bytes(self, content, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
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
                return self.send_json({"index": index, "model": model,
                                       "result": self.registry.infer(model, image, mask)})
            if parsed.path == "/api/compare":
                results = {name: self.registry.infer(name, image, mask)
                           for name in ("vit", "efficientvit")}
                return self.send_json({"index": index, "models": results})
            return self.send_json({"error": "Not found"}, 404)
        except (IndexError, ValueError):
            return self.send_json({"error": "Invalid sample index"}, 400)
        except (OSError, RuntimeError):
            return self.send_json({"error": "Could not process sample"}, 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit-checkpoint", type=Path,
                        default=os.environ.get("XLA_VIT_CHECKPOINT"))
    default_efficientvit = ROOT / "outputs" / "checkpoints" / "best_evit_seg_btxrd.pth"
    parser.add_argument("--efficientvit-checkpoint", type=Path,
                        default=os.environ.get("XLA_EFFICIENTVIT_CHECKPOINT", default_efficientvit))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        DemoHandler.dataset = TestDataset()
    except (OSError, ValueError, KeyError) as exc:
        DemoHandler.dataset_error = str(exc)
        print(f"BTXRD unavailable: {exc}", flush=True)
    DemoHandler.registry = ModelRegistry(args.vit_checkpoint,
                                        args.efficientvit_checkpoint)
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
