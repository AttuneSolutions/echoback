from __future__ import annotations

from datetime import timedelta

from echoback.db import (
    STATUS_CALLBACK_FAILED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PROCESSING,
    STATUS_QUEUED,
    STATUS_TRANSCRIBED,
    Database,
    utcnow,
)


def _add(db: Database, job_id: str, **overrides) -> None:
    kwargs = {
        "job_id": job_id,
        "job_ref": f"ref-{job_id}",
        "callback_url": "https://example.test/resume",
        "model": "small",
        "audio_path": f"/data/audio/{job_id}.wav",
    }
    kwargs.update(overrides)
    db.insert_job(**kwargs)


def test_insert_and_read(database: Database) -> None:
    _add(database, "a")
    job = database.get_job("a")
    assert job is not None
    assert job.status == STATUS_QUEUED
    assert job.job_ref == "ref-a"
    assert job.attempts == 0
    assert job.completed_at is None
    assert job.created_at.endswith("Z")
    assert database.get_job("missing") is None


def test_vocabulary_hint_round_trips(database: Database) -> None:
    _add(database, "a")
    assert database.get_job("a").vocabulary_hint is None
    _add(database, "b", vocabulary_hint="invoice, purchase order, RMA")
    assert database.get_job("b").vocabulary_hint == "invoice, purchase order, RMA"
    assert database.claim_next_queued().vocabulary_hint is None  # FIFO: job "a"


def test_init_adds_new_columns_to_a_pre_existing_db(tmp_path) -> None:
    """A jobs.db created before these columns existed must be migrated, not left broken."""
    path = tmp_path / "legacy.db"
    legacy = Database(path)
    with legacy.connect() as conn:
        conn.executescript(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY, job_ref TEXT, status TEXT NOT NULL,
                callback_url TEXT NOT NULL, model TEXT NOT NULL, audio_path TEXT,
                text TEXT, duration_ms INTEGER, error_code TEXT, error_message TEXT,
                attempts INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, completed_at TEXT
            );  -- no vocabulary_hint, no delivered_at
            INSERT INTO jobs (job_id, status, callback_url, model, attempts,
                              created_at, updated_at)
            VALUES ('legacy', 'queued', 'https://example.test/r', 'small', 0,
                    '2026-07-01T00:00:00Z', '2026-07-01T00:00:00Z');
            """
        )

    migrated = Database(path)
    migrated.init()
    legacy_job = migrated.get_job("legacy")
    assert legacy_job.vocabulary_hint is None
    assert legacy_job.delivered_at is None
    _add(migrated, "after", vocabulary_hint="back-order")
    assert migrated.get_job("after").vocabulary_hint == "back-order"
    migrated.mark_transcribed("after", text="x", duration_ms=1)
    migrated.mark_delivered("after", attempts=1)
    assert migrated.get_job("after").status == STATUS_DONE
    migrated.init()  # idempotent


def test_claim_is_fifo_and_exclusive(database: Database) -> None:
    now = utcnow()
    _add(database, "old", created_at=now - timedelta(minutes=5))
    _add(database, "new", created_at=now)

    first = database.claim_next_queued()
    assert first is not None and first.job_id == "old"
    assert first.status == STATUS_PROCESSING
    assert database.get_job("old").status == STATUS_PROCESSING

    second = database.claim_next_queued()
    assert second is not None and second.job_id == "new"

    assert database.claim_next_queued() is None


def test_queue_depth_counts_queued_and_processing(database: Database) -> None:
    _add(database, "a")
    _add(database, "b")
    assert database.queue_depth() == 2
    database.claim_next_queued()
    assert database.queue_depth() == 2
    database.mark_transcribed("a", text="hello", duration_ms=1000)
    assert database.queue_depth() == 1


def test_mark_transcribed_clears_audio_and_stamps_completion(database: Database) -> None:
    _add(database, "a")
    database.claim_next_queued()
    database.mark_transcribed("a", text="kia ora", duration_ms=30120)
    job = database.get_job("a")
    assert job.status == STATUS_TRANSCRIBED, "not done until the webhook is acked"
    assert job.text == "kia ora"
    assert job.duration_ms == 30120
    assert job.audio_path is None
    assert job.completed_at is not None
    assert job.delivered_at is None
    assert job.error_code is None


def test_mark_delivered_completes_a_transcribed_job(database: Database) -> None:
    _add(database, "a")
    database.mark_transcribed("a", text="kia ora", duration_ms=1)
    database.mark_delivered("a", attempts=2)
    job = database.get_job("a")
    assert job.status == STATUS_DONE
    assert job.attempts == 2
    assert job.delivered_at is not None


def test_mark_delivered_leaves_a_failed_job_failed(database: Database) -> None:
    """An acked *failure* webhook is delivered, but the job did not succeed."""
    _add(database, "a")
    database.mark_failed("a", code="AUDIO_DECODE_FAILED", message="nope")
    database.mark_delivered("a", attempts=1)
    job = database.get_job("a")
    assert job.status == STATUS_FAILED
    assert job.delivered_at is not None


def test_next_pending_delivery_is_fifo_and_clears_on_ack(database: Database) -> None:
    now = utcnow()
    _add(database, "old", created_at=now - timedelta(minutes=5))
    _add(database, "new", created_at=now)
    _add(database, "untouched")
    assert database.next_pending_delivery() is None, "queued jobs owe no webhook yet"

    database.mark_transcribed("new", text="second", duration_ms=1)
    database.mark_failed("old", code="AUDIO_MISSING", message="gone")
    assert database.next_pending_delivery().job_id == "old"

    database.mark_delivered("old", attempts=1)
    assert database.next_pending_delivery().job_id == "new"

    database.mark_delivered("new", attempts=1)
    assert database.next_pending_delivery() is None


def test_exhausted_callbacks_are_not_redelivered(database: Database) -> None:
    _add(database, "a")
    database.mark_transcribed("a", text="x", duration_ms=1)
    database.mark_callback_failed("a", attempts=5)
    assert database.next_pending_delivery() is None


def test_mark_failed_records_error(database: Database) -> None:
    _add(database, "a")
    database.claim_next_queued()
    database.mark_failed("a", code="AUDIO_DECODE_FAILED", message="ffmpeg said no")
    job = database.get_job("a")
    assert job.status == STATUS_FAILED
    assert job.error_code == "AUDIO_DECODE_FAILED"
    assert job.error_message == "ffmpeg said no"
    assert job.audio_path is None
    assert job.completed_at is not None


def test_callback_failed_keeps_transcript(database: Database) -> None:
    _add(database, "a")
    database.mark_transcribed("a", text="transcript", duration_ms=100)
    database.mark_callback_failed("a", attempts=5)
    job = database.get_job("a")
    assert job.status == STATUS_CALLBACK_FAILED
    assert job.attempts == 5
    assert job.text == "transcript"


def test_recover_processing_requeues(database: Database) -> None:
    _add(database, "a")
    _add(database, "b")
    database.claim_next_queued()
    assert database.recover_processing() == ["a"]
    assert database.get_job("a").status == STATUS_QUEUED
    assert database.recover_processing() == []


def test_purge_expired_only_removes_old_terminal_rows(database: Database) -> None:
    old = utcnow() - timedelta(hours=3)
    _add(database, "old-done", created_at=old)
    _add(database, "fresh-done")
    _add(database, "old-queued", created_at=old)

    database.mark_transcribed("old-done", text="x", duration_ms=1)
    database.mark_transcribed("fresh-done", text="y", duration_ms=1)
    # Backdate completion so the row falls outside the retention window.
    with database.connect() as conn:
        conn.execute(
            "UPDATE jobs SET completed_at = ?, updated_at = ? WHERE job_id = 'old-done'",
            ("2020-01-01T00:00:00Z", "2020-01-01T00:00:00Z"),
        )

    purged = database.purge_expired(retention_minutes=60)
    assert [job.job_id for job in purged] == ["old-done"]
    assert database.get_job("old-done") is None
    assert database.get_job("fresh-done") is not None
    assert database.get_job("old-queued").status == STATUS_QUEUED


def test_active_audio_paths_and_clear(database: Database) -> None:
    _add(database, "a")
    assert database.active_audio_paths() == {"/data/audio/a.wav"}
    database.clear_audio_path("a")
    assert database.active_audio_paths() == set()


def test_count_by_status(database: Database) -> None:
    _add(database, "a")
    _add(database, "b")
    database.mark_failed("b", code="X", message="y")
    assert database.count_by_status() == {STATUS_QUEUED: 1, STATUS_FAILED: 1}
