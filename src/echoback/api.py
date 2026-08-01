"""FastAPI HTTP layer."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_202_ACCEPTED,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_404_NOT_FOUND,
    HTTP_429_TOO_MANY_REQUESTS,
)

from . import __version__
from .callbacks import CallbackUrlError, validate_callback_url
from .config import Config, load_config
from .db import (
    STATUS_CALLBACK_FAILED,
    STATUS_DONE,
    STATUS_TRANSCRIBED,
    Database,
    Job,
)
from .secrets_store import Secrets, load_or_create_secrets
from .webhook import WebhookEmitter
from .whisper import WhisperEngine
from .worker import RetentionSweeper, Worker, clear_work_dirs

log = logging.getLogger("echoback.api")

UPLOAD_CHUNK = 64 * 1024
HTTP_413_TOO_LARGE = 413
WORKER_TASK_NAME = "echoback-worker"

#: Statuses whose row carries a transcript worth returning from `GET /jobs/{id}`.
TRANSCRIPT_STATUSES = (STATUS_TRANSCRIBED, STATUS_DONE, STATUS_CALLBACK_FAILED)


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass
class Components:
    config: Config
    secrets: Secrets
    database: Database
    engine: WhisperEngine
    emitter: WebhookEmitter
    worker: Worker
    sweeper: RetentionSweeper


def build_components(
    config: Config,
    *,
    secrets: Secrets | None = None,
    engine: WhisperEngine | None = None,
    emitter: WebhookEmitter | None = None,
) -> Components:
    resolved_secrets = secrets or load_or_create_secrets(
        config.secrets_path,
        api_token=config.api_token,
        webhook_secret=config.webhook_secret,
        rotate=config.rotate_secrets,
    )
    database = Database(config.db_path)
    database.init()
    config.audio_dir.mkdir(parents=True, exist_ok=True)

    resolved_engine = engine or WhisperEngine(config)
    resolved_emitter = emitter or WebhookEmitter(config, resolved_secrets.webhook_secret)
    worker = Worker(config, database, resolved_engine, resolved_emitter)
    return Components(
        config=config,
        secrets=resolved_secrets,
        database=database,
        engine=resolved_engine,
        emitter=resolved_emitter,
        worker=worker,
        sweeper=RetentionSweeper(config, database),
    )


def create_app(
    config: Config | None = None,
    *,
    secrets: Secrets | None = None,
    engine: WhisperEngine | None = None,
    emitter: WebhookEmitter | None = None,
    run_background: bool = True,
) -> FastAPI:
    resolved_config = config or load_config()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        components = build_components(
            resolved_config, secrets=secrets, engine=engine, emitter=emitter
        )
        app.state.components = components

        recovered = await asyncio.to_thread(components.database.recover_processing)
        if recovered:
            log.warning("requeued %d job(s) stranded in processing: %s", len(recovered), recovered)
        discarded = await asyncio.to_thread(clear_work_dirs, resolved_config)
        if discarded:
            log.info(
                "cleared %d temp work director%s from a previous run",
                discarded,
                "y" if discarded == 1 else "ies",
            )

        tasks: list[asyncio.Task[None]] = []
        if run_background:
            try:
                await components.engine.start()
            except Exception:  # noqa: BLE001 — serve and fail jobs loudly rather than crash-loop
                log.exception("whisper engine failed to start; jobs will fail until it recovers")
            tasks.append(asyncio.create_task(components.worker.run(), name=WORKER_TASK_NAME))
            tasks.append(asyncio.create_task(components.sweeper.run(), name="echoback-sweeper"))
        app.state.tasks = tasks
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if run_background:
                await components.engine.stop()

    app = FastAPI(
        title="Echoback",
        version=__version__,
        description="Offline voicemail transcription with a signed webhook callback.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def reject_oversized(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Cheap Content-Length gate so a huge body is refused before it is read."""
        raw_length = request.headers.get("content-length")
        if raw_length and raw_length.isdigit():
            limit = request.app.state.components.config.max_upload_bytes
            # Allow headroom for multipart boundaries around the file part itself.
            if int(raw_length) > limit + UPLOAD_CHUNK:
                message = f"request body exceeds the {limit // (1024 * 1024)} MB limit"
                return JSONResponse(
                    status_code=HTTP_413_TOO_LARGE,
                    content={"error": {"code": "PAYLOAD_TOO_LARGE", "message": message}},
                )
        return await call_next(request)

    @app.exception_handler(ApiError)
    async def _api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == HTTP_401_UNAUTHORIZED else {}
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
            headers=headers,
        )

    def components_of(request: Request) -> Components:
        return request.app.state.components  # type: ignore[no-any-return]

    async def require_auth(request: Request) -> Components:
        components = components_of(request)
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise ApiError(HTTP_401_UNAUTHORIZED, "UNAUTHORIZED", "missing bearer token")
        if not hmac.compare_digest(token.strip(), components.secrets.api_token):
            raise ApiError(HTTP_401_UNAUTHORIZED, "UNAUTHORIZED", "invalid bearer token")
        return components

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        components = components_of(request)
        depth = await asyncio.to_thread(components.database.queue_depth)
        # A worker task that died would otherwise report "idle" forever while the
        # queue silently stopped draining.
        worker_state = "stopped" if _worker_stopped(request.app) else components.worker.state
        return {
            "status": "degraded" if worker_state == "stopped" else "ok",
            "queue_depth": depth,
            "worker": worker_state,
            "model_default": components.config.model_default,
            "engine": "ready" if components.engine.is_running() else "down",
            "version": __version__,
        }

    @app.post("/jobs", status_code=HTTP_202_ACCEPTED)
    async def submit_job(
        request: Request,
        file: UploadFile | None = File(None),
        callback_url: str | None = Form(None),
        model: str | None = Form(None),
        job_ref: str | None = Form(None),
        vocabulary_hint: str | None = Form(None),
        components: Components = Depends(require_auth),
    ) -> dict[str, Any]:
        config = components.config
        if file is None or not file.filename:
            raise ApiError(HTTP_400_BAD_REQUEST, "MISSING_FILE", "a file field is required")
        resolved_url = await asyncio.to_thread(_validate_callback_url, callback_url, config)
        resolved_model = _validate_model(model, config)
        resolved_ref = _validate_job_ref(job_ref)
        resolved_hint = _validate_vocabulary_hint(vocabulary_hint, config)

        depth = await asyncio.to_thread(components.database.queue_depth)
        if depth >= config.max_queue_depth:
            raise ApiError(
                HTTP_429_TOO_MANY_REQUESTS,
                "QUEUE_FULL",
                f"queue depth {depth} has reached MAX_QUEUE_DEPTH ({config.max_queue_depth})",
            )
        # A count alone is not backpressure: MAX_QUEUE_DEPTH jobs of MAX_UPLOAD_MB each
        # would be tens of gigabytes of queued audio, so cap the bytes too.
        pending_bytes = await asyncio.to_thread(components.database.queued_bytes)
        if pending_bytes >= config.max_queue_bytes:
            raise ApiError(
                HTTP_429_TOO_MANY_REQUESTS,
                "QUEUE_FULL",
                f"queued audio ({pending_bytes // (1024 * 1024)} MB) has reached "
                f"MAX_QUEUE_MB ({config.max_queue_mb})",
            )

        job_id = str(uuid.uuid4())
        audio_path = config.audio_dir / f"{job_id}{_suffix_of(file.filename)}"
        audio_bytes = await _store_upload(file, audio_path, config.max_upload_bytes)

        try:
            await asyncio.to_thread(
                components.database.insert_job,
                job_id=job_id,
                job_ref=resolved_ref,
                callback_url=resolved_url,
                model=resolved_model,
                vocabulary_hint=resolved_hint,
                audio_path=str(audio_path),
                audio_bytes=audio_bytes,
            )
        except Exception:
            with contextlib.suppress(OSError):
                audio_path.unlink()
            raise

        components.worker.notify()
        # Never log the hint itself — it typically carries personal data.
        log.info(
            "job=%s queued model=%s ref=%s hint_chars=%d",
            job_id,
            resolved_model,
            resolved_ref,
            len(resolved_hint or ""),
        )
        return {
            "job_id": job_id,
            "job_ref": resolved_ref,
            "status": "queued",
            "status_url": config.status_url(job_id),
        }

    @app.get("/jobs/{job_id}")
    async def get_job(
        job_id: str,
        components: Components = Depends(require_auth),
    ) -> dict[str, Any]:
        job = await asyncio.to_thread(components.database.get_job, job_id)
        if job is None:
            raise ApiError(HTTP_404_NOT_FOUND, "NOT_FOUND", "unknown or purged job")
        return _job_view(job)

    return app


# ---- helpers ------------------------------------------------------------


def _worker_stopped(app: FastAPI) -> bool:
    """True when the background worker task exists but has finished — it should not."""
    return any(
        task.get_name() == WORKER_TASK_NAME and task.done()
        for task in getattr(app.state, "tasks", [])
    )


def _job_view(job: Job) -> dict[str, Any]:
    error = None
    if job.error_code:
        error = {"code": job.error_code, "message": job.error_message or ""}
    return {
        "job_id": job.job_id,
        "job_ref": job.job_ref,
        "status": job.status,
        # Any status that has a transcript exposes it: `transcribed` (webhook not yet
        # acked) and `callback_failed` (delivery gave up) both hold one, and the
        # transcript must stay fetchable here after delivery has failed.
        "text": job.text if job.status in TRANSCRIPT_STATUSES else None,
        "model": job.model,
        "duration_ms": job.duration_ms,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "delivered_at": job.delivered_at,
        "error": error,
    }


def _validate_callback_url(raw: str | None, config: Config) -> str:
    """Blocking (it resolves DNS) — callers hand this to a thread."""
    try:
        return validate_callback_url(raw, config)
    except CallbackUrlError as exc:
        raise ApiError(HTTP_400_BAD_REQUEST, exc.code, exc.message) from exc


def _validate_model(raw: str | None, config: Config) -> str:
    if raw is None or raw.strip() == "":
        return config.model_default
    value = raw.strip()
    if not config.is_model_allowed(value):
        raise ApiError(
            HTTP_400_BAD_REQUEST,
            "MODEL_NOT_ALLOWED",
            f"model {value!r} is not in the allow-list: {', '.join(config.model_allowlist)}",
        )
    return value


def _validate_job_ref(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip()
    if value == "":
        return None
    if len(value) > 256:
        raise ApiError(HTTP_400_BAD_REQUEST, "INVALID_JOB_REF", "job_ref must be ≤ 256 characters")
    if _has_control_chars(value):
        # job_ref is logged; a newline in it would let a caller forge log lines.
        raise ApiError(
            HTTP_400_BAD_REQUEST,
            "INVALID_JOB_REF",
            "job_ref must not contain control characters",
        )
    return value


def _has_control_chars(value: str, *, allow: str = "") -> bool:
    return any(ord(char) < 0x20 and char not in allow or ord(char) == 0x7F for char in value)


def _validate_vocabulary_hint(raw: str | None, config: Config) -> str | None:
    """Resolve the per-job hint, falling back to the service-wide default.

    The value is request input containing personal data: it is stored and handed to
    the engine, but never echoed back in a response, a webhook, or a log line.
    """
    value = (raw or "").strip()
    if value == "":
        return config.vocabulary_hint
    if _has_control_chars(value, allow="\n\t"):
        raise ApiError(
            HTTP_400_BAD_REQUEST,
            "INVALID_VOCABULARY_HINT",
            "vocabulary_hint must not contain control characters",
        )
    limit = config.max_vocabulary_hint_chars
    if len(value) > limit:
        raise ApiError(
            HTTP_400_BAD_REQUEST,
            "INVALID_VOCABULARY_HINT",
            f"vocabulary_hint must be ≤ {limit} characters",
        )
    return value


def _suffix_of(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if len(suffix) > 8 or not suffix[1:].isalnum():
        return ".bin"
    return suffix


async def _store_upload(file: UploadFile, dest: Path, max_bytes: int) -> int:
    """Stream the upload to disk, aborting past the size cap. Returns bytes written."""
    written = 0
    try:
        with dest.open("wb") as handle:
            while True:
                chunk = await file.read(UPLOAD_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise ApiError(
                        HTTP_413_TOO_LARGE,
                        "PAYLOAD_TOO_LARGE",
                        f"file exceeds the {max_bytes // (1024 * 1024)} MB limit",
                    )
                handle.write(chunk)
        if written == 0:
            raise ApiError(HTTP_400_BAD_REQUEST, "EMPTY_FILE", "uploaded file is empty")
        return written
    except BaseException:
        with contextlib.suppress(OSError):
            dest.unlink()
        raise
    finally:
        await file.close()
