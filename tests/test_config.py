from __future__ import annotations

import pytest

from echoback.config import DEFAULT_MODEL_ALLOWLIST, load_config


def test_defaults_match_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(
        [
            "PORT",
            "MODEL_DEFAULT",
            "MODEL_ALLOWLIST",
            "MAX_UPLOAD_MB",
            "MAX_QUEUE_DEPTH",
            "RETENTION_MINUTES",
            "WEBHOOK_ATTEMPTS",
            "WEBHOOK_BACKOFF",
            "API_TOKEN",
            "WEBHOOK_SECRET",
            "ROTATE_SECRETS",
            "DATA_DIR",
            "LOG_LEVEL",
            "VOCABULARY_HINT",
            "MAX_VOCABULARY_HINT_CHARS",
            "HOST_URL",
            "CALLBACK_ALLOWED_HOSTS",
            "MAX_QUEUE_MB",
        ]
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_config()
    assert config.port == 80
    assert config.model_default == "small"
    assert config.model_allowlist == DEFAULT_MODEL_ALLOWLIST
    assert config.max_upload_mb == 25
    assert config.max_upload_bytes == 25 * 1024 * 1024
    assert config.max_queue_depth == 1000
    assert config.retention_minutes == 60
    assert config.webhook_attempts == 8
    assert config.webhook_backoff == 2.0
    assert config.api_token is None
    assert config.webhook_secret is None
    assert config.rotate_secrets is False
    assert str(config.data_dir) == "/data"
    assert config.log_level == "info"
    assert config.vocabulary_hint is None
    assert config.max_vocabulary_hint_chars == 1000
    assert config.host_url is None
    assert config.callback_allowed_hosts == ()
    assert config.max_queue_mb == 2048
    assert str(config.db_path) == "/data/jobs.db"
    assert str(config.secrets_path) == "/data/secrets.json"


def test_allowlist_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_ALLOWLIST", "tiny, base ,small")
    monkeypatch.setenv("MODEL_DEFAULT", "base")
    config = load_config()
    assert config.model_allowlist == ("tiny", "base", "small")
    assert config.is_model_allowed("base")
    assert not config.is_model_allowed("medium")


def test_default_model_must_be_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_ALLOWLIST", "tiny")
    monkeypatch.setenv("MODEL_DEFAULT", "medium")
    with pytest.raises(ValueError, match="MODEL_DEFAULT"):
        load_config()


def test_bad_integer_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_MB", "lots")
    with pytest.raises(ValueError, match="MAX_UPLOAD_MB"):
        load_config()


@pytest.mark.parametrize("raw,expected", [("true", True), ("1", True), ("no", False), ("", False)])
def test_rotate_secrets_flag(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    monkeypatch.setenv("ROTATE_SECRETS", raw)
    assert load_config().rotate_secrets is expected


def test_vocabulary_hint_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOCABULARY_HINT", "  invoice, purchase order, RMA  ")
    assert load_config().vocabulary_hint == "invoice, purchase order, RMA"


def test_vocabulary_hint_over_the_cap_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_VOCABULARY_HINT_CHARS", "10")
    monkeypatch.setenv("VOCABULARY_HINT", "x" * 11)
    with pytest.raises(ValueError, match="VOCABULARY_HINT"):
        load_config()


def test_host_url_is_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOST_URL", "https://voicemail.example.com/")
    config = load_config()
    assert config.host_url == "https://voicemail.example.com"
    assert config.status_url("abc") == "https://voicemail.example.com/jobs/abc"


def test_status_url_falls_back_to_a_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOST_URL", raising=False)
    assert load_config().status_url("abc") == "/jobs/abc"


@pytest.mark.parametrize("bad", ["voicemail.example.com", "ftp://host", "/relative"])
def test_host_url_must_be_absolute_http(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv("HOST_URL", bad)
    with pytest.raises(ValueError, match="HOST_URL"):
        load_config()


def test_callback_allowed_hosts_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALLBACK_ALLOWED_HOSTS", " Activepieces.internal , .example.com ")
    assert load_config().callback_allowed_hosts == ("activepieces.internal", ".example.com")


def test_queue_byte_budget_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAX_QUEUE_MB", raising=False)
    config = load_config()
    assert config.max_queue_mb == 2048
    assert config.max_queue_bytes == 2048 * 1024 * 1024


def test_zero_vocabulary_hint_cap_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_VOCABULARY_HINT_CHARS", "0")
    with pytest.raises(ValueError, match="MAX_VOCABULARY_HINT_CHARS"):
        load_config()


def test_model_path_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_DIR", "/models")
    assert str(load_config().model_path("small")) == "/models/ggml-small.bin"
