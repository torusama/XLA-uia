"""Run both models over the web dataset in a persistent background job."""
import json
import sqlite3
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone


class BatchRunner:
    def __init__(self, dataset, registry, history, selection=None):
        self.dataset, self.registry, self.history = dataset, registry, history
        self.selection = selection
        self.indices = ([sample["index"] for sample in selection["included"]] if selection is not None
                        else range(len(dataset.rows) if dataset is not None else 0))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.current_id = None
        self.preview_lock = threading.Lock()
        self._preview = None
        with history.connect() as connection:
            connection.execute("UPDATE batch_jobs SET state='interrupted', "
                               "error='Backend restarted before completion' "
                               "WHERE state IN ('running','stopping')")

    def jobs(self):
        with self.history.connect() as connection:
            return [dict(row) for row in connection.execute(
                """SELECT id, state, created_at, updated_at, total, completed, current_name,
                    error, (selection IS NOT NULL AND state='complete' AND completed=total)
                    AS report_ready FROM batch_jobs ORDER BY created_at DESC LIMIT 30""")]

    def selection_summary(self):
        if self.selection is None:
            return None
        return {"source_count": self.selection["source_count"], "included_count": len(self.indices),
                "excluded_count": len(self.selection["excluded"]),
                "included_by_label": self.selection["included_by_label"]}

    def preview(self):
        """Return current input and each model's latest output, including its own sample."""
        with self.preview_lock:
            return deepcopy(self._preview)

    def _publish_preview(self, **updates):
        with self.preview_lock:
            self._preview.update(updates)
            self._preview["revision"] += 1

    def _publish_result(self, name, sample, result):
        with self.preview_lock:
            self._preview["models"][name] = {"sample": dict(sample), "result": deepcopy(result)}
            self._preview["revision"] += 1

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("A comparison job is already running")
            if not self.indices:
                raise ValueError("No test images are available for comparison")
            if not all(self.registry.statuses()[name]["ready"] for name in ("vit", "efficientvit")):
                raise ValueError("Both models must be ready")
            job_id = uuid.uuid4().hex
            now = datetime.now(timezone.utc).isoformat()
            with self.history.connect() as connection:
                connection.execute("""INSERT INTO batch_jobs
                    (id,state,created_at,updated_at,total,completed,selection) VALUES (?,?,?,?,?,?,?)""",
                                   (job_id, "running", now, now, len(self.indices), 0,
                                    json.dumps(self.selection) if self.selection is not None else None))
            self.stop_event.clear()
            self.current_id = job_id
            with self.preview_lock:
                self._preview = {"job_id": job_id, "revision": 0, "state": "running",
                                 "sample": None, "processing_model": None, "models": {}}
            self.thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
            self.thread.start()
            return job_id

    def stop(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                with self.history.connect() as connection:
                    cursor = connection.execute("""UPDATE batch_jobs SET state='stopping', updated_at=?
                        WHERE id=? AND state='running'""",
                        (datetime.now(timezone.utc).isoformat(), self.current_id))
                if cursor.rowcount:
                    self.stop_event.set()
                    self._publish_preview(state="stopping")

    def _update(self, job_id, state, current=None, error=None):
        with self.history.connect() as connection:
            connection.execute("UPDATE batch_jobs SET state=?,updated_at=?,current_name=?,error=? WHERE id=?",
                               (state, datetime.now(timezone.utc).isoformat(), current, error, job_id))

    def _run(self, job_id):
        try:
            state = "complete"
            for index in self.indices:
                if self.stop_event.is_set():
                    state = "stopped"
                    break
                sample = self.dataset.describe(index)
                self._update(job_id, "running", sample["name"])
                self._publish_preview(sample=sample, processing_model="vit")
                _, image, mask = self.dataset.read(index)
                results = {}
                for name in ("vit", "efficientvit"):
                    self._publish_preview(processing_model=name)
                    results[name] = self.registry.infer(name, image, mask)
                    self._publish_result(name, sample, results[name])
                if not all(result.get("ready") for result in results.values()):
                    raise RuntimeError("A model could not produce a result")
                self.history.record(sample, "compare", results, batch_id=job_id)
            self._finish(job_id, state)
        except (OSError, RuntimeError, ValueError, KeyError, sqlite3.Error) as exc:
            self._finish(job_id, "error", error=str(exc))

    def _finish(self, job_id, state, error=None):
        with self.lock:
            self._update(job_id, state, error=error)
            self._publish_preview(state=state, processing_model=None, error=error)
