"""Run both models over the web dataset in a persistent background job."""
import sqlite3
import threading
import uuid
from datetime import datetime, timezone


class BatchRunner:
    def __init__(self, dataset, registry, history):
        self.dataset, self.registry, self.history = dataset, registry, history
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.current_id = None
        with history.connect() as connection:
            connection.execute("UPDATE batch_jobs SET state='interrupted', "
                               "error='Backend restarted before completion' "
                               "WHERE state IN ('running','stopping')")

    def jobs(self):
        with self.history.connect() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM batch_jobs ORDER BY created_at DESC LIMIT 30")]

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("A comparison job is already running")
            if self.dataset is None:
                raise ValueError("Dataset unavailable")
            if not all(self.registry.statuses()[name]["ready"] for name in ("vit", "efficientvit")):
                raise ValueError("Both models must be ready")
            job_id = uuid.uuid4().hex
            now = datetime.now(timezone.utc).isoformat()
            with self.history.connect() as connection:
                connection.execute("INSERT INTO batch_jobs VALUES (?,?,?,?,?,?,?,?)",
                                   (job_id, "running", now, now, len(self.dataset.rows), 0, None, None))
            self.stop_event.clear()
            self.current_id = job_id
            self.thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
            self.thread.start()
            return job_id

    def stop(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                self.stop_event.set()
                self._update(self.current_id, "stopping")

    def _update(self, job_id, state, current=None, error=None):
        with self.history.connect() as connection:
            connection.execute("UPDATE batch_jobs SET state=?,updated_at=?,current_name=?,error=? WHERE id=?",
                               (state, datetime.now(timezone.utc).isoformat(), current, error, job_id))

    def _run(self, job_id):
        try:
            for index in range(len(self.dataset.rows)):
                if self.stop_event.is_set():
                    break
                sample = self.dataset.describe(index)
                self._update(job_id, "running", sample["name"])
                _, image, mask = self.dataset.read(index)
                results = {name: self.registry.infer(name, image, mask)
                           for name in ("vit", "efficientvit")}
                if not all(result.get("ready") for result in results.values()):
                    raise RuntimeError("A model could not produce a result")
                self.history.record(sample, "compare", results, batch_id=job_id)
            state = "stopped" if self.stop_event.is_set() else "complete"
            self._update(job_id, state)
        except (OSError, RuntimeError, ValueError, KeyError, sqlite3.Error) as exc:
            self._update(job_id, "error", error=str(exc))
