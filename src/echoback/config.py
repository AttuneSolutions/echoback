"""Environment-driven configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_MODEL_ALLOWLIST = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional(name: str) -> str | None:
    value = os.environ.get(name)
    return None if value is None or value.strip() == "" else value.strip()


@dataclass(frozen=True)
class Config:
    port: int = 80
    model_default: str = "small"
    model_allowlist: tuple[str, ...] = DEFAULT_MODEL_ALLOWLIST
    max_upload_mb: int = 25
    max_queue_depth: int = 1000
    max_queue_mb: int = 2048
    host_url: str | None = None
    callback_allowed_hosts: tuple[str, ...] = ()
    retention_minutes: int = 60
    webhook_attempts: int = 8
    webhook_backoff: float = 2.0
    webhook_timeout: float = 15.0
    api_token: str | None = None
    webhook_secret: str | None = None
    rotate_secrets: bool = False
    data_dir: Path = Path("/data")
    log_level: str = "info"

    # Service-wide fallback for the per-job `vocabulary_hint` form field.
    vocabulary_hint: str | None = None
    # whisper.cpp truncates the initial prompt at ~224 tokens (≈900 characters), so a
    # longer hint would lose its tail silently. Reject instead of quietly trimming.
    max_vocabulary_hint_chars: int = 1000

    # Engine wiring — not part of the documented operator surface, but
    # overridable for tests and local development.
    model_dir: Path = Path("/models")
    whisper_server_bin: str = "whisper-server"
    whisper_cli_bin: str = "whisper-cli"
    whisper_host: str = "127.0.0.1"
    whisper_port: int = 8910
    whisper_threads: int = 0  # 0 → let whisper.cpp decide
    whisper_startup_timeout: float = 180.0
    whisper_request_timeout: float = 900.0
    ffmpeg_bin: str = "ffmpeg"
    sweep_interval_seconds: float = 60.0
    extra: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_queue_bytes(self) -> int:
        return self.max_queue_mb * 1024 * 1024

    def status_url(self, job_id: str) -> str:
        """Absolute when HOST_URL is configured, path-only otherwise."""
        return f"{self.host_url}/jobs/{job_id}" if self.host_url else f"/jobs/{job_id}"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jobs.db"

    @property
    def secrets_path(self) -> Path:
        return self.data_dir / "secrets.json"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    def model_path(self, model: str) -> Path:
        return self.model_dir / f"ggml-{model}.bin"

    def is_model_allowed(self, model: str) -> bool:
        return model in self.model_allowlist


def load_config() -> Config:
    allowlist_raw = _env_optional("MODEL_ALLOWLIST")
    if allowlist_raw:
        allowlist = tuple(item.strip() for item in allowlist_raw.split(",") if item.strip())
    else:
        allowlist = DEFAULT_MODEL_ALLOWLIST
    if not allowlist:
        raise ValueError("MODEL_ALLOWLIST resolved to an empty set")

    model_default = _env_str("MODEL_DEFAULT", "small")
    if model_default not in allowlist:
        raise ValueError(
            f"MODEL_DEFAULT {model_default!r} is not in MODEL_ALLOWLIST {list(allowlist)}"
        )

    max_hint_chars = _env_int("MAX_VOCABULARY_HINT_CHARS", 1000)
    if max_hint_chars < 1:
        raise ValueError("MAX_VOCABULARY_HINT_CHARS must be at least 1")
    vocabulary_hint = _env_optional("VOCABULARY_HINT")
    if vocabulary_hint is not None and len(vocabulary_hint) > max_hint_chars:
        raise ValueError(
            f"VOCABULARY_HINT is {len(vocabulary_hint)} characters, over the "
            f"{max_hint_chars}-character limit"
        )

    host_url = _env_optional("HOST_URL")
    if host_url is not None:
        parsed = urlparse(host_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"HOST_URL must be an absolute http(s) URL, got {host_url!r}")
        host_url = host_url.rstrip("/")

    allowed_hosts_raw = _env_optional("CALLBACK_ALLOWED_HOSTS")
    allowed_hosts = (
        tuple(item.strip().lower() for item in allowed_hosts_raw.split(",") if item.strip())
        if allowed_hosts_raw
        else ()
    )

    return Config(
        port=_env_int("PORT", 80),
        model_default=model_default,
        model_allowlist=allowlist,
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 25),
        max_queue_depth=_env_int("MAX_QUEUE_DEPTH", 1000),
        max_queue_mb=_env_int("MAX_QUEUE_MB", 2048),
        host_url=host_url,
        callback_allowed_hosts=allowed_hosts,
        retention_minutes=_env_int("RETENTION_MINUTES", 60),
        webhook_attempts=_env_int("WEBHOOK_ATTEMPTS", 8),
        webhook_backoff=_env_float("WEBHOOK_BACKOFF", 2.0),
        webhook_timeout=_env_float("WEBHOOK_TIMEOUT", 15.0),
        api_token=_env_optional("API_TOKEN"),
        webhook_secret=_env_optional("WEBHOOK_SECRET"),
        rotate_secrets=_env_bool("ROTATE_SECRETS", False),
        data_dir=Path(_env_str("DATA_DIR", "/data")),
        log_level=_env_str("LOG_LEVEL", "info").lower(),
        vocabulary_hint=vocabulary_hint,
        max_vocabulary_hint_chars=max_hint_chars,
        model_dir=Path(_env_str("MODEL_DIR", "/models")),
        whisper_server_bin=_env_str("WHISPER_SERVER_BIN", "whisper-server"),
        whisper_cli_bin=_env_str("WHISPER_CLI_BIN", "whisper-cli"),
        whisper_host=_env_str("WHISPER_HOST", "127.0.0.1"),
        whisper_port=_env_int("WHISPER_PORT", 8910),
        whisper_threads=_env_int("WHISPER_THREADS", 0),
        whisper_startup_timeout=_env_float("WHISPER_STARTUP_TIMEOUT", 180.0),
        whisper_request_timeout=_env_float("WHISPER_REQUEST_TIMEOUT", 900.0),
        ffmpeg_bin=_env_str("FFMPEG_BIN", "ffmpeg"),
        sweep_interval_seconds=_env_float("SWEEP_INTERVAL_SECONDS", 60.0),
    )
