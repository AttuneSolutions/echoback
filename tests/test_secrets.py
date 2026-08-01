from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from echoback.secrets_store import load_or_create_secrets


def test_first_boot_generates_writes_and_prints_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "secrets.json"
    first = load_or_create_secrets(path)
    banner = capsys.readouterr().out

    assert first.source == "generated"
    assert first.api_token and first.webhook_secret
    assert first.api_token != first.webhook_secret
    assert "SAVE THESE — SHOWN ONCE" in banner
    assert first.api_token in banner
    assert first.webhook_secret in banner

    stored = json.loads(path.read_text())
    assert stored == {"api_token": first.api_token, "webhook_secret": first.webhook_secret}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    second = load_or_create_secrets(path)
    assert capsys.readouterr().out == ""
    assert second.source == "file"
    assert (second.api_token, second.webhook_secret) == (first.api_token, first.webhook_secret)


def test_env_secrets_take_precedence_and_write_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "secrets.json"
    result = load_or_create_secrets(path, api_token="env-token", webhook_secret="env-secret")
    assert (result.api_token, result.webhook_secret, result.source) == (
        "env-token",
        "env-secret",
        "env",
    )
    assert not path.exists()
    assert capsys.readouterr().out == ""


def test_env_overrides_one_half_of_stored_secrets(tmp_path: Path) -> None:
    path = tmp_path / "secrets.json"
    generated = load_or_create_secrets(path)
    result = load_or_create_secrets(path, api_token="env-token")
    assert result.api_token == "env-token"
    assert result.webhook_secret == generated.webhook_secret
    assert json.loads(path.read_text())["api_token"] == generated.api_token


def test_rotate_regenerates_and_reprints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "secrets.json"
    first = load_or_create_secrets(path)
    capsys.readouterr()
    rotated = load_or_create_secrets(path, rotate=True)
    out = capsys.readouterr().out
    assert rotated.api_token != first.api_token
    assert rotated.webhook_secret != first.webhook_secret
    assert rotated.api_token in out
    assert json.loads(path.read_text())["api_token"] == rotated.api_token


def test_corrupt_secrets_file_is_regenerated(tmp_path: Path) -> None:
    path = tmp_path / "secrets.json"
    path.write_text("not json at all")
    result = load_or_create_secrets(path)
    assert result.source == "generated"
    assert json.loads(path.read_text())["api_token"] == result.api_token
