"""SQLite ledger for autoresearch experiments.

Tables:
- ``experiments``: one row per experiment attempt (running, completed, failed,
  interrupted, oom, timeout). Stores the config YAML inline so the run can be
  reproduced from the ledger alone.
- ``artifacts``: file references (checkpoints, configs, reports) tied to an
  experiment.
- ``llm_cache``: response cache for LLMProposer. Key is sha256 of (model,
  system, history, phase, request).
- ``ratchet``: best-known experiment per phase, plus baseline gap.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional


DEFAULT_LEDGER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ledger.sqlite"
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phase TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    wallclock_s REAL,
    config_hash TEXT NOT NULL,
    config_yaml TEXT NOT NULL,
    parent_id INTEGER,
    git_sha TEXT,
    score REAL,
    metrics_json TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_experiments_phase ON experiments(phase);
CREATE INDEX IF NOT EXISTS idx_experiments_config_hash ON experiments(config_hash);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);
CREATE INDEX IF NOT EXISTS idx_artifacts_experiment ON artifacts(experiment_id);

CREATE TABLE IF NOT EXISTS llm_cache (
    prompt_hash TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS ratchet (
    phase TEXT PRIMARY KEY,
    best_experiment_id INTEGER,
    baseline_score REAL,
    current_score REAL,
    updated_at REAL,
    dirty INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(best_experiment_id) REFERENCES experiments(id)
);
"""


@dataclass
class ExperimentRecord:
    id: Optional[int] = None
    phase: str = ""
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    wallclock_s: Optional[float] = None
    config_hash: str = ""
    config_yaml: str = ""
    parent_id: Optional[int] = None
    git_sha: Optional[str] = None
    score: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None


class Ledger:
    def __init__(self, path: str = DEFAULT_LEDGER_PATH):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
        finally:
            conn.close()

    # --- experiments ---

    def start_experiment(self, rec: ExperimentRecord) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO experiments
                   (phase, status, started_at, config_hash, config_yaml, parent_id, git_sha, notes)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    rec.phase,
                    rec.status,
                    rec.started_at,
                    rec.config_hash,
                    rec.config_yaml,
                    rec.parent_id,
                    rec.git_sha,
                    rec.notes,
                ),
            )
            return int(cur.lastrowid)

    def finish_experiment(
        self,
        experiment_id: int,
        status: str,
        wallclock_s: float,
        score: Optional[float],
        metrics: Dict[str, Any],
        notes: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE experiments
                   SET status=?, ended_at=?, wallclock_s=?, score=?, metrics_json=?, notes=COALESCE(?, notes)
                   WHERE id=?""",
                (
                    status,
                    time.time(),
                    wallclock_s,
                    score,
                    json.dumps(metrics, default=str),
                    notes,
                    experiment_id,
                ),
            )

    def mark_orphan_running_as_interrupted(self) -> int:
        """At loop startup, any experiment still 'running' is from a dead process."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE experiments SET status='interrupted', ended_at=? WHERE status='running'",
                (time.time(),),
            )
            return cur.rowcount

    def has_completed_config(self, config_hash: str, phase: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM experiments WHERE config_hash=? AND phase=? AND status='completed' LIMIT 1",
                (config_hash, phase),
            ).fetchone()
            return row is not None

    def recent_history(self, limit: int = 50, phase: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM experiments"
        params: List[Any] = []
        if phase is not None:
            sql += " WHERE phase=?"
            params.append(phase)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def best_for_phase(
        self, phase: str, accept_partial: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Best experiment for a phase. Accepts 'partial' by default so an
        ensemble can still run when one submodel of a classify experiment
        OOM'd while others succeeded. Pass ``accept_partial=False`` to fall
        back to the strict 'completed only' filter.
        """
        statuses = ("completed", "partial") if accept_partial else ("completed",)
        placeholders = ",".join("?" * len(statuses))
        with self._connect() as conn:
            row = conn.execute(
                f"""SELECT * FROM experiments
                    WHERE phase=? AND status IN ({placeholders}) AND score IS NOT NULL
                    ORDER BY score DESC LIMIT 1""",
                (phase, *statuses),
            ).fetchone()
        return dict(row) if row else None

    # --- artifacts ---

    def add_artifact(self, experiment_id: int, kind: str, path: str, sha256: Optional[str] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO artifacts (experiment_id, kind, path, sha256) VALUES (?,?,?,?)",
                (experiment_id, kind, path, sha256),
            )

    # --- llm cache ---

    def cache_get(self, prompt_hash: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response_json FROM llm_cache WHERE prompt_hash=?",
                (prompt_hash,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["response_json"])

    def cache_put(self, prompt_hash: str, model: str, response: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (prompt_hash, model, response_json, created_at) VALUES (?,?,?,?)",
                (prompt_hash, model, json.dumps(response, default=str), time.time()),
            )

    # --- ratchet ---

    def get_ratchet(self, phase: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM ratchet WHERE phase=?", (phase,)).fetchone()
        return dict(row) if row else None

    def set_ratchet(
        self,
        phase: str,
        best_experiment_id: int,
        score: float,
        baseline: Optional[float] = None,
        dirty: bool = False,
    ) -> None:
        existing = self.get_ratchet(phase)
        baseline_score = baseline if baseline is not None else (existing["baseline_score"] if existing else score)
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO ratchet (phase, best_experiment_id, baseline_score, current_score, updated_at, dirty)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(phase) DO UPDATE SET
                     best_experiment_id=excluded.best_experiment_id,
                     baseline_score=excluded.baseline_score,
                     current_score=excluded.current_score,
                     updated_at=excluded.updated_at,
                     dirty=excluded.dirty""",
                (phase, best_experiment_id, baseline_score, score, time.time(), int(dirty)),
            )

    def mark_dirty(self, phases: Iterable[str]) -> None:
        with self._connect() as conn:
            for p in phases:
                conn.execute("UPDATE ratchet SET dirty=1 WHERE phase=?", (p,))
