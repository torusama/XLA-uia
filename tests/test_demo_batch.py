"""Offline coverage of batch completion, cancellation, and persisted progress."""
import csv
import io
import json
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pytest

from demo.backend.batch import BatchRunner
from demo.backend.comparison_report import get_report
from demo.backend.history import HistoryStore
from demo.backend.server import DemoHandler


class Samples:
    def __init__(self, count=3):
        self.rows = list(range(count))

    def describe(self, index):
        return {"index": index, "name": f"test-{index}.jpg", "label": "tumor"}

    def read(self, index):
        return b"", index, index


class Models:
    def statuses(self):
        return {name: {"ready": True} for name in ("vit", "efficientvit")}

    def infer(self, name, image, mask):
        return {"ready": True, "dice": 0.5, "iou": 1 / 3, "latency_ms": 10,
                "tp": 1, "fp": 1, "fn": 1, "measurement_config": {"device": "cpu"}}


def test_batch_runs_every_image_and_persists_pairs(tmp_path):
    history = HistoryStore(tmp_path / "runs.sqlite3")
    runner = BatchRunner(Samples(), Models(), history)
    job = runner.start()
    runner.thread.join(timeout=5)
    assert not runner.thread.is_alive()
    status = runner.jobs()[0]
    assert status["state"] == "complete" and status["completed"] == status["total"] == 3
    data = HistoryStore(history.path).query(limit=2000, batch_id=job)
    assert data["total"] == 6
    assert len({r["group_id"] for r in data["rows"]}) == 3
    assert all(s["micro_dice"] == 0.5 and s["micro_iou"] == pytest.approx(1/3)
               for s in data["summaries"])
    preview = runner.preview()
    assert preview["state"] == "complete" and preview["sample"]["index"] == 2
    assert preview["processing_model"] is None
    assert all(item["sample"]["index"] == 2 for item in preview["models"].values())


def test_batch_does_not_start_without_explicit_request(tmp_path):
    runner = BatchRunner(Samples(), Models(), HistoryStore(tmp_path / "runs.sqlite3"))
    assert runner.jobs() == []
    assert runner.thread is None
    assert runner.history.query()["total"] == 0
    assert runner.preview() is None


def test_start_again_replays_every_sample_and_keeps_previous_job(tmp_path):
    history = HistoryStore(tmp_path / "rerun.sqlite3")
    runner = BatchRunner(Samples(), Models(), history)
    first = runner.start()
    runner.thread.join(timeout=5)
    second = runner.start()
    runner.thread.join(timeout=5)

    assert first != second
    assert {job["id"]: job["completed"] for job in runner.jobs()} == {first: 3, second: 3}
    for job in (first, second):
        rows = history.query(batch_id=job)["rows"]
        assert [row["sample_index"] for row in rows] == [0, 0, 1, 1, 2, 2]


def test_report_uses_only_pairs_from_the_saved_selection(tmp_path):
    pytest.importorskip("matplotlib")
    samples = Samples()
    selection = {"source_count": 3, "included_by_label": {"tumor": 2},
                 "included": [samples.describe(0), samples.describe(2)],
                 "excluded": [dict(samples.describe(1), reason="duplicate_image_bytes")]}
    store = HistoryStore(tmp_path / "report.sqlite3")
    runner = BatchRunner(samples, Models(), store, selection)
    job = runner.start()
    runner.thread.join(timeout=5)
    assert runner.jobs()[0]["report_ready"]

    overview, files = get_report(store, job)
    assert overview["image_count"] == 2
    assert overview["summaries"]["vit"]["micro_dice"] == 0.5
    with ZipFile(io.BytesIO(files["report.zip"])) as archive:
        assert archive.testzip() is None
        assert json.loads(archive.read("selection.json")) == selection
        rows = list(csv.DictReader(io.StringIO(archive.read("per-image.csv").decode("utf-8-sig"))))
        assert {(row["sample_index"], row["model"]) for row in rows} == {
            (str(index), model) for index in (0, 2) for model in ("vit", "efficientvit")}
        assert all(archive.read(f"charts/{file}") == files[file] for file in overview["charts"])


def test_batch_stop_finishes_current_pair_and_blocks_duplicate(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class SlowModels(Models):
        def infer(self, *args):
            entered.set()
            release.wait(timeout=5)
            return super().infer(*args)

    history = HistoryStore(tmp_path / "runs.sqlite3")
    runner = BatchRunner(Samples(5), SlowModels(), history)
    job = runner.start()
    assert entered.wait(timeout=3)
    with pytest.raises(ValueError):
        runner.start()
    runner.stop()
    release.set()
    runner.thread.join(timeout=5)
    assert runner.jobs()[0]["state"] == "stopped"
    assert runner.jobs()[0]["completed"] == 1
    assert history.query(batch_id=job)["total"] == 2
    assert runner.preview()["state"] == "stopped"


def test_failed_model_does_not_commit_partial_pair(tmp_path):
    class FailedModels(Models):
        def infer(self, name, *args):
            return {"ready": False} if name == "efficientvit" else super().infer(name, *args)

    store = HistoryStore(tmp_path / "runs.sqlite3")
    runner = BatchRunner(Samples(), FailedModels(), store)
    runner.start()
    runner.thread.join(timeout=5)
    assert runner.jobs()[0]["state"] == "error"
    assert runner.jobs()[0]["completed"] == 0
    assert store.query()["total"] == 0
    assert runner.preview()["state"] == "error"


def test_batch_http_and_restart_state(tmp_path):
    history = HistoryStore(tmp_path / "runs.sqlite3")
    runner = BatchRunner(Samples(), Models(), history)

    class Handler(DemoHandler):
        dataset, registry, batch = runner.dataset, runner.registry, runner

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(root + "/api/batch") as response:
            assert json.load(response)["jobs"] == []
        assert runner.thread is None
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(root + "/api/batch/start", method="POST"))
        assert exc.value.code == 403
        with urlopen(Request(root + "/api/batch/start", method="POST",
                             headers={"X-Requested-With": "BoneScope"})) as response:
            assert response.status == 202
            job = json.load(response)["job_id"]
        runner.thread.join(timeout=5)
        with urlopen(root + "/api/batch") as response:
            result = json.load(response)
        assert result["total_samples"] == 3 and result["jobs"][0]["id"] == job
        assert result["jobs"][0]["state"] == "complete"
        assert result["preview"]["job_id"] == job
        assert result["preview"]["models"]["vit"]["sample"]["index"] == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    with history.connect() as connection:
        connection.execute("UPDATE batch_jobs SET state='running' WHERE id=?", (job,))
    reopened = BatchRunner(Samples(), Models(), HistoryStore(history.path))
    assert reopened.jobs()[0]["state"] == "interrupted"
    assert reopened.jobs()[0]["completed"] == 3
    assert reopened.preview() is None


def test_preview_keeps_each_output_with_its_own_image_and_does_not_rerun(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class PausedModels(Models):
        def __init__(self):
            self.calls = []

        def infer(self, name, image, mask):
            self.calls.append((name, image))
            if name == "efficientvit" and image == 1:
                entered.set()
                release.wait(timeout=5)
            return dict(super().infer(name, image, mask), mask_url=f"mask-{name}-{image}")

    registry = PausedModels()
    store = HistoryStore(tmp_path / "preview.sqlite3")
    runner = BatchRunner(Samples(2), registry, store)
    runner.start()
    try:
        assert entered.wait(timeout=3)
        preview = runner.preview()
        assert preview["sample"]["index"] == 1
        assert preview["processing_model"] == "efficientvit"
        assert preview["models"]["vit"]["sample"]["index"] == 1
        assert preview["models"]["vit"]["result"]["mask_url"] == "mask-vit-1"
        assert preview["models"]["efficientvit"]["sample"]["index"] == 0
        assert preview["models"]["efficientvit"]["result"]["mask_url"] == "mask-efficientvit-0"
        calls = list(registry.calls)
        preview["models"]["vit"]["result"]["mask_url"] = "mutated-by-reader"
        assert runner.preview()["models"]["vit"]["result"]["mask_url"] == "mask-vit-1"
        assert registry.calls == calls
        assert store.query()["total"] == 2  # Only the first complete pair is committed.
    finally:
        release.set()
        runner.thread.join(timeout=5)
