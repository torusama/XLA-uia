"""Persist real inference measurements and query recent demo history."""
import hashlib
import json
import math
import sqlite3
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

METRIC_KEYS = ("latency_ms", "dice", "iou", "params_m", "gflops", "cpu_seconds", "cpu_cores",
               "rss_peak_mb", "rss_delta_mb", "gpu_peak_mb", "gpu_delta_mb", "tp", "fp", "fn")


class HistoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session = uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL,
                group_id TEXT NOT NULL, mode TEXT NOT NULL, created_at TEXT NOT NULL,
                model TEXT NOT NULL, config_id TEXT NOT NULL, sample_index INTEGER,
                sample_name TEXT, sample_label TEXT, config TEXT NOT NULL, metrics TEXT NOT NULL)""")
            connection.execute("CREATE INDEX IF NOT EXISTS runs_session ON runs(session, id)")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
            if "batch_id" not in columns:
                connection.execute("ALTER TABLE runs ADD COLUMN batch_id TEXT")
            connection.execute("CREATE INDEX IF NOT EXISTS runs_batch ON runs(batch_id, id)")
            connection.execute("""CREATE TABLE IF NOT EXISTS batch_jobs (
                id TEXT PRIMARY KEY, state TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, total INTEGER NOT NULL, completed INTEGER NOT NULL,
                current_name TEXT, error TEXT)""")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def record(self, sample, mode, results, batch_id=None):
        group = uuid.uuid4().hex
        with self.connect() as connection:
            for model, result in results.items():
                if not result.get("ready"):
                    continue
                config = result.get("measurement_config", {})
                config_id = hashlib.sha256(json.dumps(
                    {"model": model, **config}, sort_keys=True).encode()).hexdigest()[:16]
                metrics = {key: result.get(key) for key in METRIC_KEYS}
                cursor = connection.execute("""INSERT INTO runs
                    (session,group_id,mode,created_at,model,config_id,sample_index,
                     sample_name,sample_label,config,metrics,batch_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (self.session, group, mode, result.get("inference_started_at") or
                     datetime.now(timezone.utc).isoformat(), model, config_id, sample["index"],
                     sample["name"], sample.get("label"), json.dumps(config), json.dumps(metrics), batch_id))
                result["history_id"] = cursor.lastrowid
            if batch_id:
                connection.execute("UPDATE batch_jobs SET completed=completed+1, updated_at=? WHERE id=?",
                                   (datetime.now(timezone.utc).isoformat(), batch_id))
        return group

    def query(self, limit=200, session="all", mode="compare", hours=0, batch_id=""):
        if not 1 <= limit <= 2000 or mode not in ("compare", "single", "all") or not 0 <= hours <= 8760:
            raise ValueError("Invalid history filter")
        clauses, values = [], []
        if session != "all":
            clauses.append("session = ?")
            values.append(self.session if session == "current" else session)
        if mode != "all":
            clauses.append("mode = ?")
            values.append(mode)
        if hours:
            clauses.append("created_at >= ?")
            values.append((datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat())
        if batch_id:
            clauses.append("batch_id = ?")
            values.append(batch_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM runs" + where, values).fetchone()[0]
            raw = connection.execute("SELECT * FROM runs" + where + " ORDER BY id DESC LIMIT ?",
                                     [*values, limit]).fetchall()
            sessions = connection.execute("SELECT session,MIN(created_at) AS started_at,COUNT(*) AS count "
                                          "FROM runs GROUP BY session ORDER BY started_at DESC").fetchall()
        rows = []
        grouped = defaultdict(list)
        for record in reversed(raw):
            row = dict(record)
            row["config"] = json.loads(row["config"])
            row.update(json.loads(row.pop("metrics")))
            rows.append(row)
            grouped[row["config_id"]].append(row)
        summaries = []
        for config_id, runs in grouped.items():
            summary = {"config_id": config_id, "model": runs[0]["model"],
                       "config": runs[0]["config"], "count": len(runs),
                       "unique_samples": len({row["sample_index"] for row in runs})}
            for key in METRIC_KEYS:
                numbers = [r[key] for r in runs if isinstance(r.get(key), (int, float))
                           and math.isfinite(r[key])]
                summary[f"mean_{key}"] = sum(numbers) / len(numbers) if numbers else None
            times = sorted(r["latency_ms"] for r in runs if (r.get("latency_ms") or 0) > 0)
            summary["median_ms"] = ((times[(len(times)-1)//2] + times[len(times)//2]) / 2) if times else None
            summary["p95_ms"] = times[math.ceil(0.95 * len(times))-1] if times else None
            summary["throughput_ips"] = len(times) * 1000 / sum(times) if times else None
            counted = [r for r in runs if all(numeric is not None for numeric in
                                               (r.get("tp"), r.get("fp"), r.get("fn")))]
            if counted:
                tp, fp, fn = (sum(r[key] for r in counted) for key in ("tp", "fp", "fn"))
                summary["micro_dice"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
                summary["micro_iou"] = tp / (tp + fp + fn) if tp + fp + fn else 1.0
            summaries.append(summary)
        return {"rows": rows, "summaries": summaries, "total": total, "shown": len(rows),
                "current_session": self.session, "sessions": [dict(row) for row in sessions]}
