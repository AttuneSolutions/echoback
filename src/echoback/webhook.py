"""Outbound signed webhook delivery."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from . import __version__
from .config import Config
from .db import STATUS_DONE, STATUS_TRANSCRIBED, Job

log = logging.getLogger("echoback.webhook")

USER_AGENT = f"echoback/{__version__}"


def sign(secret: str, body: bytes) -> str:
    """``sha256=<hex>`` HMAC of the raw request body."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify(secret: str, body: bytes, signature: str) -> bool:
    return hmac.compare_digest(sign(secret, body), signature or "")


def build_payload(job: Job) -> dict[str, Any]:
    """The webhook body. `error` is populated only on the failure path.

    A job being delivered is still ``transcribed`` — it becomes ``done`` only once
    this payload is acknowledged — but the receiver's contract is unchanged: a
    successful transcription always reports `"status": "done"`.
    """
    succeeded = job.status in (STATUS_TRANSCRIBED, STATUS_DONE)
    error: dict[str, str] | None = None
    if not succeeded:
        error = {
            "code": job.error_code or "TRANSCRIPTION_FAILED",
            "message": job.error_message or "transcription failed",
        }
    return {
        "job_id": job.job_id,
        "job_ref": job.job_ref,
        "status": STATUS_DONE if succeeded else "failed",
        "text": job.text if succeeded else None,
        "model": job.model,
        "duration_ms": job.duration_ms if succeeded else None,
        "completed_at": job.completed_at,
        "error": error,
    }


def encode_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass
class DeliveryResult:
    delivered: bool
    attempts: int
    last_status: int | None = None
    last_error: str | None = None

    @property
    def summary(self) -> str:
        if self.last_status is not None:
            return f"HTTP {self.last_status}"
        return self.last_error or "unknown error"


SleepFn = Callable[[float], Awaitable[None]]
ClientFactory = Callable[[], httpx.AsyncClient]


class WebhookEmitter:
    """POSTs a job result to its callback URL, retrying with exponential backoff.

    Backoff between attempts doubles from ``WEBHOOK_BACKOFF``: with the defaults
    (5 attempts, base 2s) retries are delayed 2s, 4s, 8s and 16s — 30s of waiting
    across roughly a minute of wall clock before the job is marked
    ``callback_failed``.
    """

    def __init__(
        self,
        config: Config,
        secret: str,
        *,
        client_factory: ClientFactory | None = None,
        sleep: SleepFn | None = None,
    ) -> None:
        self._config = config
        self._secret = secret
        self._client_factory = client_factory or self._default_client_factory
        self._sleep = sleep or asyncio.sleep

    def _default_client_factory(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._config.webhook_timeout, follow_redirects=False)

    def backoff_for(self, attempt: int) -> float:
        """Delay before ``attempt`` (1-based); attempt 1 is immediate."""
        return self._config.webhook_backoff * (2 ** (attempt - 2)) if attempt > 1 else 0.0

    async def deliver(self, job: Job) -> DeliveryResult:
        payload = build_payload(job)
        body = encode_payload(payload)
        headers = {
            "Content-Type": "application/json",
            "X-Signature": sign(self._secret, body),
            "X-Job-Id": job.job_id,
            "User-Agent": USER_AGENT,
        }
        max_attempts = max(1, self._config.webhook_attempts)
        result = DeliveryResult(delivered=False, attempts=0)
        host = urlparse(job.callback_url).hostname or "the callback host"

        async with self._client_factory() as client:
            for attempt in range(1, max_attempts + 1):
                delay = self.backoff_for(attempt)
                if delay:
                    await self._sleep(delay)
                result.attempts = attempt
                try:
                    response = await client.post(job.callback_url, content=body, headers=headers)
                except httpx.HTTPError as exc:
                    result.last_status = None
                    # Only the exception type and the host: several httpx errors embed
                    # the full URL in their message, and a callback URL's path is a
                    # capability secret that must not reach the logs.
                    result.last_error = f"{type(exc).__name__} contacting {host}"
                    log.warning(
                        "job=%s webhook attempt %d/%d failed: %s",
                        job.job_id,
                        attempt,
                        max_attempts,
                        result.last_error,
                    )
                    continue

                result.last_status = response.status_code
                result.last_error = None
                if 200 <= response.status_code < 300:
                    result.delivered = True
                    log.info(
                        "job=%s webhook delivered on attempt %d (HTTP %d)",
                        job.job_id,
                        attempt,
                        response.status_code,
                    )
                    return result
                log.warning(
                    "job=%s webhook attempt %d/%d returned HTTP %d",
                    job.job_id,
                    attempt,
                    max_attempts,
                    response.status_code,
                )

        log.error(
            "job=%s webhook exhausted %d attempts (last: %s)",
            job.job_id,
            result.attempts,
            result.summary,
        )
        return result
