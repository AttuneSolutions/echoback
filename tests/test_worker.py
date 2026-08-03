from __future__ import annotations

import asyncio
import dataclasses
import json
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from echoback.config import Config
from echoback.db import (
    STATUS_CALLBACK_FAILED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_TRANSCRIBED,
    Database,
    utcnow,
)
from echoback.webhook import WebhookEmitter, verify
from echoback.whisper import TranscriptionError
from echoback.worker import RetentionSweeper, Worker, clear_work_dirs

SECRET = "worker-secret"
CALLBACK = "https://receiver.test/resume/xyz"


class FakeEngine:
    """Stands in for whisper.cpp: records calls, can stall or raise."""

    def __init__(
        self, text: str = "kia ora, it's Aroha", *, error: Exception | None = None
    ) -> None:
        self.text = text
        self.error = error
        self.calls: list[tuple[Path, str, str | None]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.gate: asyncio.Event | None = None

    def is_running(self) -> bool:
        return True

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def transcribe(
        self, wav_path: Path, model: str, vocabulary_hint: str | None = None
    ) -> str:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            assert wav_path.exists(), "worker must hand a real file to the engine"
            self.calls.append((wav_path, model, vocabulary_hint))
            if self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(0)
            if self.error is not None:
                raise self.error
            return f"{self.text} [{model}]"
        finally:
            self.in_flight -= 1


class Receiver:
    """Collects webhook deliveries; `statuses` drives the responses it returns."""

    def __init__(self, statuses: list[int] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.statuses = statuses or []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status = self.statuses.pop(0) if self.statuses else 200
        return httpx.Response(status)

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]

    def factory(self):
        return lambda: httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


async def _noop_sleep(_: float) -> None:
    return None


@pytest.fixture
def receiver() -> Receiver:
    return Receiver()


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


def make_worker(config: Config, database: Database, engine, receiver: Receiver) -> Worker:
    emitter = WebhookEmitter(config, SECRET, client_factory=receiver.factory(), sleep=_noop_sleep)
    return Worker(config, database, engine, emitter)


def enqueue(
    config: Config,
    database: Database,
    make_wav,
    *,
    job_id: str,
    model: str = "small",
    vocabulary_hint: str | None = None,
):
    audio_path = config.audio_dir / f"{job_id}.wav"
    make_wav(audio_path)
    return database.insert_job(
        job_id=job_id,
        job_ref=f"ref-{job_id}",
        callback_url=CALLBACK,
        model=model,
        vocabulary_hint=vocabulary_hint,
        audio_path=str(audio_path),
    )


# ---- happy path ---------------------------------------------------------


async def test_processes_job_deletes_audio_and_delivers_signed_webhook(
    config: Config, database: Database, engine: FakeEngine, receiver: Receiver, make_wav
) -> None:
    job = enqueue(config, database, make_wav, job_id="job-1")
    worker = make_worker(config, database, engine, receiver)

    assert await worker.run_once() is True

    stored = database.get_job("job-1")
    assert stored.status == STATUS_DONE
    assert stored.text == "kia ora, it's Aroha [small]"
    assert stored.duration_ms == 1000
    assert stored.audio_path is None
    assert stored.attempts == 1
    assert stored.delivered_at is not None
    assert not Path(job.audio_path).exists(), "uploaded audio must be deleted after processing"
    assert list(config.audio_dir.iterdir()) == [], "no temp artefacts left behind"

    assert len(receiver.requests) == 1
    request = receiver.requests[0]
    assert verify(SECRET, request.content, request.headers["x-signature"])
    body = receiver.bodies[0]
    assert body["status"] == "done"
    assert body["job_ref"] == "ref-job-1"
    assert body["text"] == "kia ora, it's Aroha [small]"
    assert body["error"] is None
    assert worker.state == "idle"


async def test_run_once_on_empty_queue(
    config: Config, database: Database, engine: FakeEngine, receiver: Receiver
) -> None:
    worker = make_worker(config, database, engine, receiver)
    assert await worker.run_once() is False
    assert receiver.requests == []


async def test_model_override_is_passed_to_engine(
    config: Config, database: Database, engine: FakeEngine, receiver: Receiver, make_wav
) -> None:
    enqueue(config, database, make_wav, job_id="job-1", model="medium")
    worker = make_worker(config, database, engine, receiver)
    await worker.run_once()
    assert engine.calls[0][1] == "medium"
    assert receiver.bodies[0]["model"] == "medium"


async def test_vocabulary_hint_is_passed_to_engine_but_not_webhooked(
    config: Config, database: Database, engine: FakeEngine, receiver: Receiver, make_wav
) -> None:
    hint = "invoice, purchase order, back-order; Acme Holdings"
    enqueue(config, database, make_wav, job_id="job-1", vocabulary_hint=hint)
    worker = make_worker(config, database, engine, receiver)
    await worker.run_once()

    assert engine.calls[0][2] == hint
    assert "vocabulary_hint" not in receiver.bodies[0]
    # `Acme` appears nowhere but the hint, so finding it would mean the hint leaked.
    assert b"Acme" not in receiver.requests[0].content


async def test_no_vocabulary_hint_reaches_the_engine_as_none(
    config: Config, database: Database, engine: FakeEngine, receiver: Receiver, make_wav
) -> None:
    enqueue(config, database, make_wav, job_id="job-1")
    await make_worker(config, database, engine, receiver).run_once()
    assert engine.calls[0][2] is None


# ---- delivery acknowledgement -------------------------------------------


async def test_job_is_not_done_until_the_receiver_acks(
    config: Config, database: Database, engine: FakeEngine, make_wav
) -> None:
    """The status the receiver's 2xx unlocks: transcribed while in flight, done after."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(database.get_job("job-1").status)
        return httpx.Response(200)

    receiver = Receiver()
    receiver.handler = handler  # type: ignore[method-assign]
    enqueue(config, database, make_wav, job_id="job-1")
    await make_worker(config, database, engine, receiver).run_once()

    assert seen == [STATUS_TRANSCRIBED], "status must not be done while delivery is in flight"
    assert database.get_job("job-1").status == STATUS_DONE


async def test_a_2xx_other_than_200_still_counts_as_acked(
    config: Config, database: Database, engine: FakeEngine, make_wav
) -> None:
    receiver = Receiver(statuses=[204])
    enqueue(config, database, make_wav, job_id="job-1")
    await make_worker(config, database, engine, receiver).run_once()
    assert database.get_job("job-1").status == STATUS_DONE


async def test_non_2xx_is_retried_and_then_gives_up_without_marking_done(
    config: Config, database: Database, engine: FakeEngine, make_wav
) -> None:
    receiver = Receiver(statuses=[500, 302, 404, 500, 418])
    five = dataclasses.replace(config, webhook_attempts=5)
    enqueue(config, database, make_wav, job_id="job-1")
    await make_worker(five, database, engine, receiver).run_once()

    stored = database.get_job("job-1")
    assert len(receiver.requests) == 5
    assert stored.status == STATUS_CALLBACK_FAILED
    assert stored.delivered_at is None


async def test_result_stranded_by_a_crash_is_redelivered_not_retranscribed(
    config: Config, database: Database, engine: FakeEngine, receiver: Receiver, make_wav
) -> None:
    """A process that died between transcription and the ack leaves a `transcribed` row."""
    enqueue(config, database, make_wav, job_id="job-1")
    database.claim_next_queued()
    database.mark_transcribed("job-1", text="kia ora, it's Aroha", duration_ms=1000)

    worker = make_worker(config, database, engine, receiver)
    assert await worker.run_once() is True

    assert engine.calls == [], "the transcript already exists; only the webhook was owed"
    assert receiver.bodies[0]["text"] == "kia ora, it's Aroha"
    assert receiver.bodies[0]["status"] == "done"
    assert database.get_job("job-1").status == STATUS_DONE
    assert await worker.run_once() is False, "an acked job is not delivered twice"


async def test_stranded_failure_webhook_is_also_redelivered(
    config: Config, database: Database, engine: FakeEngine, receiver: Receiver, make_wav
) -> None:
    enqueue(config, database, make_wav, job_id="job-1")
    database.claim_next_queued()
    database.mark_failed("job-1", code="AUDIO_DECODE_FAILED", message="ffmpeg said no")

    worker = make_worker(config, database, engine, receiver)
    assert await worker.run_once() is True

    body = receiver.bodies[0]
    assert body["status"] == "failed"
    assert body["error"]["code"] == "AUDIO_DECODE_FAILED"
    stored = database.get_job("job-1")
    assert stored.status == STATUS_FAILED, "an acked failure is still a failure"
    assert stored.delivered_at is not None
    assert await worker.run_once() is False


async def test_stranded_delivery_is_handled_before_new_work(
    config: Config, database: Database, engine: FakeEngine, receiver: Receiver, make_wav
) -> None:
    enqueue(config, database, make_wav, job_id="stranded")
    database.claim_next_queued()
    database.mark_transcribed("stranded", text="older result", duration_ms=1)
    enqueue(config, database, make_wav, job_id="fresh")

    worker = make_worker(config, database, engine, receiver)
    await worker.run_once()
    await worker.run_once()
    assert [body["job_id"] for body in receiver.bodies] == ["stranded", "fresh"]


# ---- failure paths ------------------------------------------------------


async def test_undecodable_audio_yields_failure_webhook(
    config: Config, database: Database, engine: FakeEngine, receiver: Receiver
) -> None:
    audio_path = config.audio_dir / "corrupt.bin"
    audio_path.write_bytes(b"definitely not audio")
    database.insert_job(
        job_id="job-bad",
        job_ref=None,
        callback_url=CALLBACK,
        model="small",
        audio_path=str(audio_path),
    )
    worker = make_worker(config, database, engine, receiver)
    await worker.run_once()

    stored = database.get_job("job-bad")
    assert stored.status == STATUS_FAILED
    assert stored.error_code == "AUDIO_DECODE_FAILED"
    assert not audio_path.exists()
    assert engine.calls == []

    body = receiver.bodies[0]
    assert body["status"] == "failed"
    assert body["text"] is None
    assert body["duration_ms"] is None
    assert body["error"]["code"] == "AUDIO_DECODE_FAILED"


async def test_missing_audio_file_fails_fast(
    config: Config, database: Database, engine: FakeEngine, receiver: Receiver
) -> None:
    database.insert_job(
        job_id="job-gone",
        job_ref=None,
        callback_url=CALLBACK,
        model="small",
        audio_path=str(config.audio_dir / "vanished.wav"),
    )
    worker = make_worker(config, database, engine, receiver)
    await worker.run_once()
    assert database.get_job("job-gone").error_code == "AUDIO_MISSING"
    assert receiver.bodies[0]["status"] == "failed"


async def test_engine_error_marks_job_failed_with_engine_code(
    config: Config, database: Database, receiver: Receiver, make_wav
) -> None:
    engine = FakeEngine(error=TranscriptionError("no weights", code="MODEL_UNAVAILABLE"))
    enqueue(config, database, make_wav, job_id="job-1")
    worker = make_worker(config, database, engine, receiver)
    await worker.run_once()

    stored = database.get_job("job-1")
    assert stored.status == STATUS_FAILED
    assert stored.error_code == "MODEL_UNAVAILABLE"
    assert receiver.bodies[0]["error"]["code"] == "MODEL_UNAVAILABLE"


async def test_unexpected_engine_exception_is_contained(
    config: Config, database: Database, receiver: Receiver, make_wav
) -> None:
    engine = FakeEngine(error=RuntimeError("boom"))
    enqueue(config, database, make_wav, job_id="job-1")
    worker = make_worker(config, database, engine, receiver)
    await worker.run_once()
    stored = database.get_job("job-1")
    assert stored.status == STATUS_FAILED
    assert stored.error_code == "TRANSCRIPTION_FAILED"
    assert "boom" in stored.error_message


async def test_callback_exhaustion_marks_callback_failed_but_keeps_transcript(
    config: Config, database: Database, engine: FakeEngine, make_wav
) -> None:
    receiver = Receiver(statuses=[500, 500, 500, 500, 500])
    five = dataclasses.replace(config, webhook_attempts=5)
    enqueue(config, database, make_wav, job_id="job-1")
    worker = make_worker(five, database, engine, receiver)
    await worker.run_once()

    stored = database.get_job("job-1")
    assert len(receiver.requests) == 5
    assert stored.status == STATUS_CALLBACK_FAILED
    assert stored.attempts == 5
    assert stored.text == "kia ora, it's Aroha [small]"


async def test_an_unexpected_error_does_not_kill_the_loop(
    config: Config, database: Database, engine: FakeEngine, receiver: Receiver, make_wav
) -> None:
    """A dead worker task would stop the queue while /health still looked fine."""
    worker = make_worker(config, database, engine, receiver)
    calls = {"n": 0}
    original = worker.run_once

    async def flaky() -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("something unexpected")
        return await original()

    worker.run_once = flaky  # type: ignore[method-assign]
    enqueue(config, database, make_wav, job_id="job-1")

    import echoback.worker as worker_mod

    monkey = worker_mod.ERROR_BACKOFF_SECONDS
    worker_mod.ERROR_BACKOFF_SECONDS = 0.0
    try:
        task = asyncio.create_task(worker.run())
        for _ in range(500):
            if database.get_job("job-1").status == STATUS_DONE:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        worker_mod.ERROR_BACKOFF_SECONDS = monkey

    assert calls["n"] >= 2, "the loop must keep running after an unexpected error"
    assert database.get_job("job-1").status == STATUS_DONE


# ---- ordering & concurrency --------------------------------------------


async def test_queue_is_fifo_with_concurrency_one(
    config: Config, database: Database, engine: FakeEngine, receiver: Receiver, make_wav
) -> None:
    engine.gate = asyncio.Event()
    for index in (1, 2, 3):
        enqueue(config, database, make_wav, job_id=f"job-{index}")
    worker = make_worker(config, database, engine, receiver)

    task = asyncio.create_task(worker.run())
    await asyncio.sleep(0.05)
    assert engine.in_flight == 1, "only one transcription may be in flight"
    assert database.get_job("job-2").status == "queued"
    engine.gate.set()

    for _ in range(500):
        if len(receiver.requests) >= 3:
            break
        await asyncio.sleep(0.02)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert engine.max_in_flight == 1
    assert [body["job_id"] for body in receiver.bodies] == ["job-1", "job-2", "job-3"]


# ---- retention ---------------------------------------------------------


async def test_sweeper_purges_expired_rows_only(
    config: Config, database: Database, make_wav
) -> None:
    database.insert_job(
        job_id="old",
        job_ref=None,
        callback_url=CALLBACK,
        model="small",
        audio_path="/nonexistent",
        created_at=utcnow() - timedelta(hours=5),
    )
    database.mark_transcribed("old", text="ancient", duration_ms=1)
    with database.connect() as conn:
        conn.execute("UPDATE jobs SET completed_at = '2020-01-01T00:00:00Z' WHERE job_id = 'old'")
    database.insert_job(
        job_id="fresh",
        job_ref=None,
        callback_url=CALLBACK,
        model="small",
        audio_path="/nonexistent",
    )
    database.mark_transcribed("fresh", text="recent", duration_ms=1)

    sweeper = RetentionSweeper(config, database)
    assert await sweeper.sweep_once() == 1
    assert database.get_job("old") is None
    assert database.get_job("fresh") is not None


async def test_sweeper_removes_orphaned_audio(config: Config, database: Database, make_wav) -> None:
    orphan = make_wav(config.audio_dir / "orphan.wav")
    import os
    import time

    stale = time.time() - 7200
    os.utime(orphan, (stale, stale))
    referenced = make_wav(config.audio_dir / "referenced.wav")
    os.utime(referenced, (stale, stale))
    database.insert_job(
        job_id="live",
        job_ref=None,
        callback_url=CALLBACK,
        model="small",
        audio_path=str(referenced),
    )

    await RetentionSweeper(config, database).sweep_once()
    assert not orphan.exists()
    assert referenced.exists()


async def test_sweeper_leaves_work_dirs_alone(config: Config, database: Database, make_wav) -> None:
    """A long transcription's work dir must not be swept out from under it."""
    import os
    import time

    work_dir = config.audio_dir / "echoback-inflight"
    work_dir.mkdir()
    make_wav(work_dir / "normalized.wav")
    stale = time.time() - 7200
    os.utime(work_dir, (stale, stale))

    await RetentionSweeper(config, database).sweep_once()
    assert (work_dir / "normalized.wav").exists()


async def test_sweeper_keeps_recent_audio(config: Config, database: Database, make_wav) -> None:
    recent = make_wav(config.audio_dir / "recent.wav")
    await RetentionSweeper(config, database).sweep_once()
    assert recent.exists()


async def test_retention_window_is_configurable(
    config: Config, database: Database, make_wav
) -> None:
    zero_retention = dataclasses.replace(config, retention_minutes=0)
    database.insert_job(
        job_id="job-1",
        job_ref=None,
        callback_url=CALLBACK,
        model="small",
        audio_path="/nonexistent",
    )
    database.mark_transcribed("job-1", text="x", duration_ms=1)
    assert await RetentionSweeper(zero_retention, database).sweep_once() == 1


async def test_clear_work_dirs_drops_leftovers_from_a_dead_process(
    config: Config, make_wav
) -> None:
    stale = config.audio_dir / "echoback-abc123"
    stale.mkdir()
    make_wav(stale / "normalized.wav")
    upload = make_wav(config.audio_dir / "job-1.wav")

    assert clear_work_dirs(config) == 1
    assert not stale.exists()
    assert upload.exists(), "a pending upload is not a work dir and must survive"
    assert clear_work_dirs(config) == 0
