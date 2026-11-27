"""Offline coverage of batch completion, cancellation, and persisted progress."""
import json
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from demo.backend.batch import BatchRunner
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


def test_batch_does_not_start_without_explicit_request(tmp_path):
    runner = BatchRunner(Samples(), Models(), HistoryStore(tmp_path / "runs.sqlite3"))
    assert runner.jobs() == []
    assert runner.thread is None
    assert runner.history.query()["total"] == 0


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
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    with history.connect() as connection:
        connection.execute("UPDATE batch_jobs SET state='running' WHERE id=?", (job,))
    reopened = BatchRunner(Samples(), Models(), HistoryStore(history.path))
    assert reopened.jobs()[0]["state"] == "interrupted"
    assert reopened.jobs()[0]["completed"] == 3
