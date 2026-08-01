"""Single-worker transcription loop and the retention sweeper."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import tempfile
import time
from pathlib import Path

from . import audio as audio_mod
from .config import Config
from .db import Database, Job
from .webhook import WebhookEmitter
from .whisper import TranscriptionError, WhisperEngine

log = logging.getLogger("echoback.worker")

ORPHAN_GRACE_SECONDS = 3600
WORK_DIR_PREFIX = "echoback-"
ERROR_BACKOFF_SECONDS = 5.0


class Worker:
    """Drains the queue one job at a time, then emits each result webhook."""

    def __init__(
        self,
        config: Config,
        database: Database,
        engine: WhisperEngine,
        emitter: WebhookEmitter,
    ) -> None:
        self._config = config
        self._db = database
        self._engine = engine
        self._emitter = emitter
        self._wakeup = asyncio.Event()
        self._busy = False

    @property
    def state(self) -> str:
        return "busy" if self._busy else "idle"

    def notify(self) -> None:
        """Signal that a job was enqueued so the loop wakes immediately."""
        self._wakeup.set()

    async def run(self) -> None:
        log.info("worker loop started")
        try:
            while True:
                try:
                    processed = await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — the loop must outlive any one job
                    # Without this the task would die and the queue would stop draining
                    # while /health still looked healthy.
                    log.exception("worker iteration failed; retrying shortly")
                    await asyncio.sleep(ERROR_BACKOFF_SECONDS)
                    continue
                if processed:
                    continue
                self._wakeup.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._wakeup.wait(), timeout=5.0)
        except asyncio.CancelledError:
            log.info("worker loop stopped")
            raise

    async def run_once(self) -> bool:
        """Claim and fully handle one job. Returns False when there is nothing to do."""
        stranded = await asyncio.to_thread(self._db.next_pending_delivery)
        if stranded is not None:
            # A previous process died between transcription and the ack. The
            # result already exists; only the webhook is owed.
            log.warning("job=%s resuming undelivered result", stranded.job_id)
            await self._deliver(stranded.job_id)
            return True

        job = await asyncio.to_thread(self._db.claim_next_queued)
        if job is None:
            return False
        self._busy = True
        try:
            await self._transcribe(job)
            await self._deliver(job.job_id)
        finally:
            self._busy = False
        return True

    # ---- steps ----------------------------------------------------------

    async def _transcribe(self, job: Job) -> None:
        started = time.monotonic()
        source = Path(job.audio_path) if job.audio_path else None
        if source is None or not source.exists():
            log.error("job=%s audio file is missing", job.job_id)
            await asyncio.to_thread(
                self._db.mark_failed,
                job.job_id,
                code="AUDIO_MISSING",
                message="uploaded audio was not found on disk",
            )
            return

        work_dir = Path(tempfile.mkdtemp(prefix=WORK_DIR_PREFIX, dir=str(source.parent)))
        normalized = work_dir / "normalized.wav"
        try:
            try:
                await audio_mod.normalize(source, normalized, ffmpeg_bin=self._config.ffmpeg_bin)
            except audio_mod.AudioDecodeError as exc:
                log.warning("job=%s audio decode failed", job.job_id)
                await asyncio.to_thread(
                    self._db.mark_failed,
                    job.job_id,
                    code="AUDIO_DECODE_FAILED",
                    message=str(exc),
                )
                return

            duration = await asyncio.to_thread(audio_mod.duration_ms, normalized)
            try:
                text = await self._engine.transcribe(normalized, job.model, job.vocabulary_hint)
            except TranscriptionError as exc:
                log.warning("job=%s transcription failed (%s)", job.job_id, exc.code)
                await asyncio.to_thread(
                    self._db.mark_failed, job.job_id, code=exc.code, message=str(exc)
                )
                return
            except Exception as exc:  # noqa: BLE001 — never let one job kill the loop
                log.exception("job=%s unexpected transcription error", job.job_id)
                await asyncio.to_thread(
                    self._db.mark_failed,
                    job.job_id,
                    code="TRANSCRIPTION_FAILED",
                    message=f"{type(exc).__name__}: {exc}",
                )
                return

            await asyncio.to_thread(
                self._db.mark_transcribed, job.job_id, text=text, duration_ms=duration
            )
            log.info(
                "job=%s transcribed model=%s audio_ms=%s elapsed_ms=%d",
                job.job_id,
                job.model,
                duration,
                round((time.monotonic() - started) * 1000),
            )
        finally:
            # Audio never outlives the processing step.
            _remove_tree(work_dir)
            _remove_file(source)
            await asyncio.to_thread(self._db.clear_audio_path, job.job_id)

    async def _deliver(self, job_id: str) -> None:
        job = await asyncio.to_thread(self._db.get_job, job_id)
        if job is None:
            log.warning("job=%s vanished before webhook delivery", job_id)
            return
        result = await self._emitter.deliver(job)
        if result.delivered:
            # Only an acknowledged webhook completes a successful job.
            await asyncio.to_thread(self._db.mark_delivered, job_id, attempts=result.attempts)
            log.info("job=%s delivered after %d attempt(s)", job_id, result.attempts)
        else:
            await asyncio.to_thread(self._db.mark_callback_failed, job_id, attempts=result.attempts)
            log.error(
                "job=%s callback_failed after %d attempts (last: %s)",
                job_id,
                result.attempts,
                result.summary,
            )


class RetentionSweeper:
    """Purges expired job rows and any stray audio files."""

    def __init__(self, config: Config, database: Database) -> None:
        self._config = config
        self._db = database

    async def run(self) -> None:
        log.info("retention sweeper started (%d minutes)", self._config.retention_minutes)
        try:
            while True:
                try:
                    await self.sweep_once()
                except Exception:  # noqa: BLE001 — a bad sweep must not stop future sweeps
                    log.exception("retention sweep failed")
                await asyncio.sleep(self._config.sweep_interval_seconds)
        except asyncio.CancelledError:
            log.info("retention sweeper stopped")
            raise

    async def sweep_once(self) -> int:
        purged = await asyncio.to_thread(
            self._db.purge_expired, retention_minutes=self._config.retention_minutes
        )
        for job in purged:
            if job.audio_path:
                _remove_file(Path(job.audio_path))
        if purged:
            log.info("purged %d expired job rows", len(purged))
        await asyncio.to_thread(self._sweep_orphan_audio)
        return len(purged)

    def _sweep_orphan_audio(self) -> None:
        audio_dir = self._config.audio_dir
        if not audio_dir.exists():
            return
        active = self._db.active_audio_paths()
        cutoff = time.time() - ORPHAN_GRACE_SECONDS
        for entry in audio_dir.iterdir():
            if str(entry) in active:
                continue
            if entry.is_dir() and entry.name.startswith(WORK_DIR_PREFIX):
                # A work dir belongs to the job currently in flight, and its mtime stops
                # moving once the normalised audio is written — a transcription running
                # longer than the grace period would have its input deleted underneath
                # it. Leftovers from a dead process are cleared at startup instead.
                continue
            try:
                if entry.stat().st_mtime > cutoff:
                    continue
            except OSError:
                continue
            if entry.is_dir():
                _remove_tree(entry)
            else:
                _remove_file(entry)
            log.info("removed orphaned audio artefact %s", entry.name)


def clear_work_dirs(config: Config) -> int:
    """Drop temp work directories left by a process that died mid-job.

    Called at startup only: with a single worker, nothing is in flight at boot, so
    any `echoback-*` directory belongs to a previous process and its normalised
    audio (which can be several times the upload size) should go immediately rather
    than wait for the orphan grace period.
    """
    audio_dir = config.audio_dir
    if not audio_dir.exists():
        return 0
    removed = 0
    for entry in audio_dir.iterdir():
        if entry.is_dir() and entry.name.startswith(WORK_DIR_PREFIX):
            _remove_tree(entry)
            removed += 1
    return removed


def _remove_file(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


def _remove_tree(path: Path) -> None:
    with contextlib.suppress(OSError):
        shutil.rmtree(path, ignore_errors=True)
