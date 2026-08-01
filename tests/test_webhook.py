from __future__ import annotations

import dataclasses
import json

import httpx
import pytest

from echoback import __version__
from echoback.config import Config
from echoback.db import Job
from echoback.webhook import WebhookEmitter, build_payload, encode_payload, sign, verify

SECRET = "s3cr3t"


def make_job(**overrides) -> Job:
    base = {
        "job_id": "b3f1c2a7-9d4e-4f8a-8c11-2e6d5a0b1234",
        "job_ref": "vm-2026-08-01-0042",
        "status": "done",
        "callback_url": "https://receiver.test/resume/abc",
        "model": "small",
        "audio_path": None,
        "text": "Hi, it's Aroha calling from ...",
        "duration_ms": 30120,
        "error_code": None,
        "error_message": None,
        "attempts": 0,
        "created_at": "2026-08-01T09:15:03Z",
        "updated_at": "2026-08-01T09:15:11Z",
        "completed_at": "2026-08-01T09:15:11Z",
    }
    base.update(overrides)
    return Job(**base)


def client_factory(handler):
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


class SleepSpy:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


# ---- signing ------------------------------------------------------------


def test_signature_shape_and_verification() -> None:
    body = b'{"job_id":"x"}'
    signature = sign(SECRET, body)
    assert signature.startswith("sha256=")
    assert len(signature) == len("sha256=") + 64
    assert verify(SECRET, body, signature)
    assert not verify(SECRET, body + b" ", signature)  # tampered body
    assert not verify("other-secret", body, signature)
    assert not verify(SECRET, body, "")


# ---- payload ------------------------------------------------------------


def test_success_payload_matches_spec() -> None:
    payload = build_payload(make_job())
    assert payload == {
        "job_id": "b3f1c2a7-9d4e-4f8a-8c11-2e6d5a0b1234",
        "job_ref": "vm-2026-08-01-0042",
        "status": "done",
        "text": "Hi, it's Aroha calling from ...",
        "model": "small",
        "duration_ms": 30120,
        "completed_at": "2026-08-01T09:15:11Z",
        "error": None,
    }


def test_failure_payload_matches_spec() -> None:
    job = make_job(
        status="failed",
        text=None,
        duration_ms=None,
        error_code="AUDIO_DECODE_FAILED",
        error_message="ffmpeg could not decode the uploaded file",
        completed_at="2026-08-01T09:15:07Z",
    )
    payload = build_payload(job)
    assert payload["status"] == "failed"
    assert payload["text"] is None
    assert payload["duration_ms"] is None
    assert payload["error"] == {
        "code": "AUDIO_DECODE_FAILED",
        "message": "ffmpeg could not decode the uploaded file",
    }


def test_callback_failed_job_reports_as_failed_with_transcript_withheld() -> None:
    payload = build_payload(make_job(status="callback_failed"))
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "TRANSCRIPTION_FAILED"


# ---- delivery -----------------------------------------------------------


async def test_delivers_on_first_attempt_with_headers(config: Config) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text="ok")

    sleeper = SleepSpy()
    emitter = WebhookEmitter(config, SECRET, client_factory=client_factory(handler), sleep=sleeper)
    job = make_job()
    result = await emitter.deliver(job)

    assert result.delivered is True
    assert result.attempts == 1
    assert result.last_status == 200
    assert sleeper.delays == []

    request = seen[0]
    assert str(request.url) == job.callback_url
    assert request.headers["content-type"] == "application/json"
    assert request.headers["x-job-id"] == job.job_id
    assert request.headers["user-agent"] == f"echoback/{__version__}"
    assert verify(SECRET, request.content, request.headers["x-signature"])
    assert json.loads(request.content) == build_payload(job)
    assert request.content == encode_payload(build_payload(job))


async def test_retries_with_exponential_backoff_then_succeeds(config: Config) -> None:
    statuses = [500, 502, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(statuses.pop(0))

    sleeper = SleepSpy()
    emitter = WebhookEmitter(config, SECRET, client_factory=client_factory(handler), sleep=sleeper)
    result = await emitter.deliver(make_job())

    assert result.delivered is True
    assert result.attempts == 3
    assert sleeper.delays == [2.0, 4.0]


async def test_exhausts_five_attempts_then_reports_failure(config: Config) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    sleeper = SleepSpy()
    emitter = WebhookEmitter(config, SECRET, client_factory=client_factory(handler), sleep=sleeper)
    result = await emitter.deliver(make_job())

    assert calls == 5
    assert result.delivered is False
    assert result.attempts == 5
    assert result.last_status == 503
    assert result.summary == "HTTP 503"
    assert sleeper.delays == [2.0, 4.0, 8.0, 16.0]


async def test_transport_errors_are_retried(config: Config) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(204)

    emitter = WebhookEmitter(
        config, SECRET, client_factory=client_factory(handler), sleep=SleepSpy()
    )
    result = await emitter.deliver(make_job())
    assert result.delivered is True
    assert result.attempts == 3


async def test_failure_summary_never_leaks_the_callback_url(config: Config) -> None:
    """Callback URLs are capability URLs; several httpx errors embed them in str(exc)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed to connect to {request.url}", request=request)

    emitter = WebhookEmitter(
        config, SECRET, client_factory=client_factory(handler), sleep=SleepSpy()
    )
    job = make_job(callback_url="https://receiver.test/resume/super-secret-token")
    result = await emitter.deliver(job)

    assert result.delivered is False
    assert "super-secret-token" not in result.summary
    assert "ConnectError" in result.summary
    assert "receiver.test" in result.summary, "the host is useful and not secret"


async def test_network_failure_summary(config: Config) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    emitter = WebhookEmitter(
        config, SECRET, client_factory=client_factory(handler), sleep=SleepSpy()
    )
    result = await emitter.deliver(make_job())
    assert result.delivered is False
    assert result.last_status is None
    assert "ConnectTimeout" in result.summary


@pytest.mark.parametrize("attempt,expected", [(1, 0.0), (2, 2.0), (3, 4.0), (4, 8.0), (5, 16.0)])
async def test_backoff_schedule(config: Config, attempt: int, expected: float) -> None:
    emitter = WebhookEmitter(config, SECRET, sleep=SleepSpy())
    assert emitter.backoff_for(attempt) == expected


async def test_attempt_count_is_configurable(config: Config) -> None:
    tight = dataclasses.replace(config, webhook_attempts=2)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    emitter = WebhookEmitter(
        tight, SECRET, client_factory=client_factory(handler), sleep=SleepSpy()
    )
    result = await emitter.deliver(make_job())
    assert calls == 2
    assert result.attempts == 2
    assert result.delivered is False
