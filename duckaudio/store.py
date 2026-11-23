from __future__ import annotations

import json
import re
from pathlib import Path
import sqlite3
import threading
import time

from .errors import AudioError

TERMINAL = frozenset(("finished", "cancelled", "failed", "expired"))


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o750)
        self.clips = self.directory / "clips"
        self.clips.mkdir(mode=0o750, exist_ok=True)
        self.tmp = self.directory / "tmp"
        self.tmp.mkdir(mode=0o700, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.directory / "audio.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
          PRAGMA journal_mode=WAL;
          PRAGMA foreign_keys=ON;
          CREATE TABLE IF NOT EXISTS clips (
            id TEXT PRIMARY KEY, key TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
            filename TEXT NOT NULL, sha256 TEXT NOT NULL, duration REAL NOT NULL,
            bytes INTEGER NOT NULL, source TEXT NOT NULL, created REAL NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}');
          CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, request_id TEXT UNIQUE, fingerprint TEXT NOT NULL,
            clip_id TEXT NOT NULL, task_id TEXT NOT NULL, state TEXT NOT NULL,
            created REAL NOT NULL, updated REAL NOT NULL, data TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT, time REAL NOT NULL,
            kind TEXT NOT NULL, data TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        self.db.commit()
        # The application already owns its instance lock; these cannot be active files.
        for path in self.tmp.iterdir():
            if path.is_file() and not path.is_symlink() and re.fullmatch(r"[0-9a-f]{32}\.(?:wav|txt|log|upload)", path.name):
                path.unlink()
        referenced = {row[0] for row in self.db.execute("SELECT filename FROM clips")}
        for path in self.clips.iterdir():
            if path.name not in referenced and re.fullmatch(r"[0-9a-f]{32}\.wav", path.name) and not path.is_symlink():
                path.unlink()
        # A daemon restart must never resurrect an old movement-associated cue.
        for row in self.db.execute("SELECT data FROM jobs WHERE state NOT IN ('finished','cancelled','failed','expired')").fetchall():
            job = json.loads(row[0])
            job.update(state="failed", error="音频服务重启，原播放未恢复", error_code="service_restarted", updated=time.time())
            self.save_job(job)
        self.event("service_started", {})

    def _clip(self, row):
        if row is None:
            raise AudioError("音频资源不存在", "clip_not_found", 404)
        item = dict(row)
        item["metadata"] = json.loads(item["metadata"])
        return item

    def list_clips(self):
        with self.lock:
            return [self._clip(r) for r in self.db.execute("SELECT * FROM clips ORDER BY created DESC")]

    def get_clip(self, identifier):
        with self.lock:
            return self._clip(self.db.execute("SELECT * FROM clips WHERE id=? OR key=?", (identifier, identifier)).fetchone())

    def add_clip(self, clip):
        with self.lock:
            try:
                self.db.execute("INSERT INTO clips VALUES (?,?,?,?,?,?,?,?,?,?)", (
                    clip["id"], clip["key"], clip["name"], clip["filename"], clip["sha256"],
                    clip["duration"], clip["bytes"], clip["source"], clip["created"], json.dumps(clip["metadata"], ensure_ascii=False)))
                self.db.commit()
            except sqlite3.IntegrityError as exc:
                self.db.rollback()
                raise AudioError("音频名称键已存在，请换一个资源键", "duplicate_key", 409) from exc
        return clip

    def rename_clip(self, identifier, name, key=None):
        with self.lock:
            clip = self.get_clip(identifier)
            try:
                self.db.execute("UPDATE clips SET name=?, key=? WHERE id=?", (name, key or clip["key"], clip["id"]))
                self.db.commit()
            except sqlite3.IntegrityError as exc:
                self.db.rollback()
                raise AudioError("资源键已存在", "duplicate_key", 409) from exc
            return self.get_clip(clip["id"])

    def delete_clip(self, identifier):
        with self.lock:
            clip = self.get_clip(identifier)
            self.db.execute("DELETE FROM clips WHERE id=?", (clip["id"],))
            self.db.commit()
            (self.clips / clip["filename"]).unlink(missing_ok=True)

    def total_bytes(self):
        with self.lock:
            return self.db.execute("SELECT COALESCE(SUM(bytes),0) FROM clips").fetchone()[0]

    def save_job(self, job):
        with self.lock:
            self.db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state, updated=excluded.updated, data=excluded.data", (
                job["id"], job.get("request_id"), job["fingerprint"], job["clip_id"], job["task_id"], job["state"],
                job["created"], job["updated"], json.dumps(job, ensure_ascii=False)))
            if job["state"] in TERMINAL:
                self.db.execute("""DELETE FROM jobs WHERE id IN (
                    SELECT id FROM jobs WHERE state IN ('finished','cancelled','failed','expired')
                    ORDER BY created DESC LIMIT -1 OFFSET 10000)""")
            self.db.commit()

    def get_job(self, identifier):
        with self.lock:
            row = self.db.execute("SELECT data FROM jobs WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise AudioError("播放任务不存在", "job_not_found", 404)
            return json.loads(row[0])

    def find_request(self, request_id):
        if not request_id:
            return None
        with self.lock:
            row = self.db.execute("SELECT data FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def list_jobs(self, task_id=None, limit=100):
        with self.lock:
            sql = "SELECT data FROM jobs"
            args = []
            if task_id is not None:
                sql += " WHERE task_id=?"
                args.append(task_id)
            sql += " ORDER BY created DESC LIMIT ?"
            args.append(min(max(int(limit), 1), 200))
            return [json.loads(r[0]) for r in self.db.execute(sql, args)]

    def event(self, kind, data):
        with self.lock:
            self.db.execute("INSERT INTO events(time,kind,data) VALUES(?,?,?)", (time.time(), kind, json.dumps(data, ensure_ascii=False)))
            # Bounded event history; the last 10,000 terminal jobs retain idempotency.
            self.db.execute("DELETE FROM events WHERE seq < (SELECT COALESCE(MAX(seq),0)-10000 FROM events)")
            self.db.commit()

    def events(self, after=0, limit=200):
        with self.lock:
            return [{"seq": r[0], "time": r[1], "kind": r[2], "data": json.loads(r[3])}
                    for r in self.db.execute("SELECT seq,time,kind,data FROM events WHERE seq>? ORDER BY seq LIMIT ?", (after, min(limit, 500)))]

    def get_setting(self, key, default=None):
        with self.lock:
            row = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else default

    def set_setting(self, key, value):
        with self.lock:
            self.db.execute("INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))
            self.db.commit()

    def close(self):
        with self.lock:
            self.db.close()
