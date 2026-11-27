"""Persistent history, filtered aggregates, and concurrent recording checks."""
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from demo.backend.history import HistoryStore
from demo.backend.server import DemoHandler

SAMPLE = {"index": 3, "name": "example.jpg", "label": "tumor"}


def measurement(latency=10.0, threshold=0.5):
    return {"ready": True, "mask_url": "not persisted", "latency_ms": latency,
            "dice": 0.7, "iou": 0.5, "params_m": 1.0, "gflops": 2.0,
            "measurement_config": {"checkpoint_sha256": "abc", "threshold": threshold,
                                   "device": "cpu"}}


def test_history_survives_restart_and_separates_configurations(tmp_path):
    path = tmp_path / "history.sqlite3"
    store = HistoryStore(path)
    for latency in (10, 20, 30):
        store.record(SAMPLE, "compare", {"vit": measurement(latency)})
    store.record(SAMPLE, "single", {"vit": measurement(50)})
    store.record(SAMPLE, "compare", {"vit": measurement(40, threshold=0.8)})
    reopened = HistoryStore(path)
    data = reopened.query()
    assert data["total"] == data["shown"] == 4
    assert len(data["summaries"]) == 2
    old = next(row for row in data["summaries"] if row["config"]["threshold"] == 0.5)
    assert old["mean_latency_ms"] == old["median_ms"] == 20
    assert old["p95_ms"] == 30 and old["throughput_ips"] == 50
    assert reopened.query(session="current")["total"] == 0
    assert reopened.query(session=store.session, mode="all")["total"] == 5
    assert data["rows"][0].get("mask_url") is None
    assert reopened.query(limit=2)["shown"] == 2
    with pytest.raises(ValueError):
        reopened.query(limit=0)


def test_concurrent_pairs_commit_without_losing_records(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")

    def record(_):
        store.record(SAMPLE, "compare", {"vit": measurement(), "efficientvit": measurement()})

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(record, range(12)))
    rows = store.query()["rows"]
    assert len(rows) == 24
    assert len({row["group_id"] for row in rows}) == 12
    for group in {row["group_id"] for row in rows}:
        assert {row["model"] for row in rows if row["group_id"] == group} == {"vit", "efficientvit"}


def test_history_api_csv_and_invalid_filter(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    store.record(SAMPLE, "compare", {"vit": measurement()})

    class Handler(DemoHandler):
        history = store

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(tmp_path)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(root + "/api/history") as response:
            assert json.load(response)["total"] == 1
        with urlopen(root + "/api/history.csv") as response:
            text = response.read().decode("utf-8-sig")
            assert "example.jpg" in text and "latency_ms" in text
        with pytest.raises(HTTPError) as exc:
            urlopen(root + "/api/history?limit=5000")
        assert exc.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
