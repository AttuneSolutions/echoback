"""First-boot secret bootstrap."""

from __future__ import annotations

import json
import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("echoback.secrets")

_TOKEN_BYTES = 32


@dataclass(frozen=True)
class Secrets:
    api_token: str
    webhook_secret: str
    source: str  # "env" | "generated" | "file"

    @property
    def generated(self) -> bool:
        return self.source == "generated"


def _banner(secrets_obj: Secrets) -> str:
    return (
        "\n================ SAVE THESE — SHOWN ONCE ================\n"
        f"API_TOKEN:      {secrets_obj.api_token}"
        "         (use as: Authorization: Bearer <token>)\n"
        f"WEBHOOK_SECRET: {secrets_obj.webhook_secret}"
        "         (verify X-Signature with this)\n"
        "========================================================\n"
    )


def load_or_create_secrets(
    path: Path,
    *,
    api_token: str | None = None,
    webhook_secret: str | None = None,
    rotate: bool = False,
) -> Secrets:
    """Resolve both secrets.

    Env-supplied values win outright and nothing is written to disk. Otherwise the
    secrets file is read, or created (mode 600) and printed once.
    """
    if api_token and webhook_secret:
        log.info("using API_TOKEN and WEBHOOK_SECRET from the environment")
        return Secrets(api_token=api_token, webhook_secret=webhook_secret, source="env")

    stored = _read_file(path) if path.exists() and not rotate else None

    if stored is None:
        generated = Secrets(
            api_token=api_token or secrets.token_urlsafe(_TOKEN_BYTES),
            webhook_secret=webhook_secret or secrets.token_urlsafe(_TOKEN_BYTES),
            source="generated",
        )
        _write_file(path, generated)
        if rotate:
            log.warning("ROTATE_SECRETS was set — previous secrets have been replaced")
        print(_banner(generated), flush=True)  # noqa: T201 — deliberate one-time output
        return generated

    # Env may still override one half; that half is not persisted.
    resolved = Secrets(
        api_token=api_token or stored.api_token,
        webhook_secret=webhook_secret or stored.webhook_secret,
        source="file",
    )
    log.info("loaded secrets from %s", path)
    return resolved


def _read_file(path: Path) -> Secrets | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Secrets(
            api_token=payload["api_token"],
            webhook_secret=payload["webhook_secret"],
            source="file",
        )
    except (OSError, ValueError, KeyError, TypeError):
        log.warning("secrets file at %s is unreadable — regenerating", path)
        return None


def _write_file(path: Path, secrets_obj: Secrets) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = json.dumps(
        {
            "api_token": secrets_obj.api_token,
            "webhook_secret": secrets_obj.webhook_secret,
        },
        indent=2,
    )
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    os.replace(tmp, path)
    os.chmod(path, 0o600)
