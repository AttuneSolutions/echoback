"""SQLite job store — also the queue.

All functions here are synchronous; async callers wrap them in ``asyncio.to_thread``.
A fresh connection is opened per operation, which is cheap for SQLite and keeps the
API reader and the single worker writer from sharing connection state.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_TRANSCRIBED = "transcribed"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CALLBACK_FAILED = "callback_failed"

TERMINAL_STATUSES = (STATUS_DONE, STATUS_FAILED, STATUS_CALLBACK_FAILED)

#: Statuses whose result still owes the caller a webhook. ``transcribed`` always
#: does; ``failed`` does until its failure webhook is acknowledged, which is why
#: both are matched against a NULL ``delivered_at`` rather than status alone.
AWAITING_DELIVERY_STATUSES = (STATUS_TRANSCRIBED, STATUS_FAILED)

#: Statuses the retention sweeper may purge. ``transcribed`` is not terminal, but a
#: row still sitting there past the retention window belongs to a delivery nobody is
#: coming back for, and retention does not allow a transcript to outlive the window.
PURGEABLE_STATUSES = (*TERMINAL_STATUSES, STATUS_TRANSCRIBED)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    job_ref       TEXT,
    status        TEXT NOT NULL,
    callback_url  TEXT NOT NULL,
    model         TEXT NOT NULL,
    vocabulary_hint TEXT,
    audio_path    TEXT,
    text          TEXT,
    duration_ms   INTEGER,
    error_code    TEXT,
    error_message TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    completed_at  TEXT,
    delivered_at  TEXT,
    audio_bytes   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs (status, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_completed_at ON jobs (completed_at);
"""


# Columns added after v1 shipped. `CREATE TABLE IF NOT EXISTS` leaves an existing
# jobs.db untouched, so each one is also applied as an idempotent ALTER.
ADDED_COLUMNS = (
    ("vocabulary_hint", "TEXT"),
    ("delivered_at", "TEXT"),
    ("audio_bytes", "INTEGER"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    present = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
    for name, decl in ADDED_COLUMNS:
        if name not in present:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {decl}")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    """ISO-8601 UTC with a trailing Z, seconds-and-below preserved."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Job:
    job_id: str
    job_ref: str | None
    status: str
    callback_url: str
    model: str
    audio_path: str | None
    text: str | None
    duration_ms: int | None
    error_code: str | None
    error_message: str | None
    attempts: int
    created_at: str
    updated_at: str
    completed_at: str | None
    # Trail the required fields so existing positional/keyword construction still works.
    vocabulary_hint: str | None = None
    delivered_at: str | None = None
    audio_bytes: int | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Job:
        # Ignore columns this build does not know about, so an older image reading a
        # newer /data/jobs.db degrades instead of raising on every read.
        known = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: row[key] for key in row.keys() if key in known})


class Database:
    def __init__(self, path: Path | str, *, busy_timeout_ms: int = 5000) -> None:
        self.path = Path(path)
        self.busy_timeout_ms = busy_timeout_ms

    # ---- plumbing -------------------------------------------------------

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=self.busy_timeout_ms / 1000, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            _migrate(conn)

    # ---- writes ---------------------------------------------------------

    def insert_job(
        self,
        *,
        job_id: str,
        job_ref: str | None,
        callback_url: str,
        model: str,
        audio_path: str,
        vocabulary_hint: str | None = None,
        audio_bytes: int | None = None,
        created_at: datetime | None = None,
    ) -> Job:
        stamp = iso(created_at or utcnow())
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, job_ref, status, callback_url, model, vocabulary_hint,
                                  audio_path, audio_bytes, attempts, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    job_id,
                    job_ref,
                    STATUS_QUEUED,
                    callback_url,
                    model,
                    vocabulary_hint,
                    audio_path,
                    audio_bytes,
                    stamp,
                    stamp,
                ),
            )
        job = self.get_job(job_id)
        assert job is not None
        return job

    def claim_next_queued(self) -> Job | None:
        """Move the oldest queued job to ``processing`` and return it (FIFO)."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    """
                    SELECT * FROM jobs WHERE status = ?
                    ORDER BY created_at, rowid LIMIT 1
                    """,
                    (STATUS_QUEUED,),
                ).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return None
                stamp = iso(utcnow())
                conn.execute(
                    "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ? AND status = ?",
                    (STATUS_PROCESSING, stamp, row["job_id"], STATUS_QUEUED),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        job = Job.from_row(row)
        job.status = STATUS_PROCESSING
        job.updated_at = stamp
        return job

    def mark_transcribed(self, job_id: str, *, text: str, duration_ms: int | None) -> None:
        """Transcription succeeded. The job is not ``done`` until the webhook is acked."""
        stamp = iso(utcnow())
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                   SET status = ?, text = ?, duration_ms = ?, error_code = NULL,
                       error_message = NULL, audio_path = NULL,
                       completed_at = ?, updated_at = ?
                 WHERE job_id = ?
                """,
                (STATUS_TRANSCRIBED, text, duration_ms, stamp, stamp, job_id),
            )

    def mark_delivered(self, job_id: str, *, attempts: int) -> None:
        """The receiver returned 2xx. A successful job only reaches ``done`` here.

        A job that failed to transcribe keeps its ``failed`` status — the ack is
        recorded by ``delivered_at`` alone, so it is no longer redelivered.
        """
        stamp = iso(utcnow())
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                   SET status = CASE WHEN status = ? THEN ? ELSE status END,
                       attempts = ?, delivered_at = ?, updated_at = ?
                 WHERE job_id = ?
                """,
                (STATUS_TRANSCRIBED, STATUS_DONE, attempts, stamp, stamp, job_id),
            )

    def mark_failed(self, job_id: str, *, code: str, message: str) -> None:
        stamp = iso(utcnow())
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                   SET status = ?, error_code = ?, error_message = ?, audio_path = NULL,
                       completed_at = ?, updated_at = ?
                 WHERE job_id = ?
                """,
                (STATUS_FAILED, code, message, stamp, stamp, job_id),
            )

    def mark_callback_failed(self, job_id: str, *, attempts: int) -> None:
        stamp = iso(utcnow())
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, attempts = ?, updated_at = ? WHERE job_id = ?",
                (STATUS_CALLBACK_FAILED, attempts, stamp, job_id),
            )

    def clear_audio_path(self, job_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET audio_path = NULL, updated_at = ? WHERE job_id = ?",
                (iso(utcnow()), job_id),
            )

    def recover_processing(self) -> list[str]:
        """Reset jobs stranded in ``processing`` by a crash back to ``queued``."""
        stamp = iso(utcnow())
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT job_id FROM jobs WHERE status = ?", (STATUS_PROCESSING,)
            ).fetchall()
            if rows:
                conn.execute(
                    "UPDATE jobs SET status = ?, updated_at = ? WHERE status = ?",
                    (STATUS_QUEUED, stamp, STATUS_PROCESSING),
                )
        return [row["job_id"] for row in rows]

    def purge_expired(self, *, retention_minutes: int, now: datetime | None = None) -> list[Job]:
        """Delete terminal job rows older than the retention window."""
        cutoff = iso((now or utcnow()) - timedelta(minutes=retention_minutes))
        placeholders = ",".join("?" for _ in PURGEABLE_STATUSES)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute(
                    f"""
                    SELECT * FROM jobs
                     WHERE status IN ({placeholders})
                       AND COALESCE(completed_at, updated_at) <= ?
                    """,
                    (*PURGEABLE_STATUSES, cutoff),
                ).fetchall()
                if rows:
                    conn.executemany(
                        "DELETE FROM jobs WHERE job_id = ?",
                        [(row["job_id"],) for row in rows],
                    )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return [Job.from_row(row) for row in rows]

    # ---- reads ----------------------------------------------------------

    def next_pending_delivery(self) -> Job | None:
        """The oldest job whose result was never acknowledged.

        Normally empty: delivery follows transcription inside the same worker turn.
        A row shows up here when the process died between the two, and would
        otherwise be purged with its result undelivered.
        """
        placeholders = ",".join("?" for _ in AWAITING_DELIVERY_STATUSES)
        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM jobs
                 WHERE status IN ({placeholders}) AND delivered_at IS NULL
                 ORDER BY created_at, rowid LIMIT 1
                """,
                AWAITING_DELIVERY_STATUSES,
            ).fetchone()
        return Job.from_row(row) if row else None

    def get_job(self, job_id: str) -> Job | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return Job.from_row(row) if row else None

    def queue_depth(self) -> int:
        """Jobs not yet transcribed — queued plus the one in flight."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE status IN (?, ?)",
                (STATUS_QUEUED, STATUS_PROCESSING),
            ).fetchone()
        return int(row["n"])

    def queued_bytes(self) -> int:
        """Bytes of audio on disk for jobs not yet transcribed (backpressure)."""
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(audio_bytes), 0) AS total FROM jobs
                 WHERE status IN (?, ?) AND audio_bytes IS NOT NULL
                """,
                (STATUS_QUEUED, STATUS_PROCESSING),
            ).fetchone()
        return int(row["total"])

    def count_by_status(self) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        return {row["status"]: int(row["n"]) for row in rows}

    def active_audio_paths(self) -> set[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT audio_path FROM jobs WHERE audio_path IS NOT NULL"
            ).fetchall()
        return {row["audio_path"] for row in rows}
