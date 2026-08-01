from __future__ import annotations

import struct
import textwrap
import wave
from pathlib import Path

import pytest

from echoback.config import Config
from echoback.db import Database
from echoback.secrets_store import Secrets

API_TOKEN = "test-api-token"
WEBHOOK_SECRET = "test-webhook-secret"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "data"
    (path / "audio").mkdir(parents=True)
    return path


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    path = tmp_path / "models"
    path.mkdir()
    for name in ("small", "medium", "tiny"):
        (path / f"ggml-{name}.bin").write_bytes(b"fake-weights")
    return path


@pytest.fixture
def config(data_dir: Path, model_dir: Path, fake_ffmpeg: Path) -> Config:
    return Config(
        port=8080,
        data_dir=data_dir,
        model_dir=model_dir,
        ffmpeg_bin=str(fake_ffmpeg),
        webhook_backoff=2.0,
        sweep_interval_seconds=0.05,
    )


#: Hostnames the tests may use in a callback_url, and what they "resolve" to. Real
#: DNS is never consulted in the suite — see the autouse fixture below.
FAKE_DNS = {
    "receiver.test": ["203.0.113.10"],
    "flows.example.com": ["198.51.100.20"],
    "activepieces.internal": ["10.4.0.9"],  # a private LAN receiver is allowed
    "metadata.evil.test": ["169.254.169.254"],  # DNS pointing at cloud metadata
    "sneaky.evil.test": ["203.0.113.10", "127.0.0.1"],  # one good answer, one blocked
}


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve callback hosts from FAKE_DNS; anything else fails to resolve."""

    def resolve(host: str) -> list[str]:
        try:
            return FAKE_DNS[host]
        except KeyError:
            raise OSError(f"fake DNS has no record for {host!r}") from None

    monkeypatch.setattr("echoback.callbacks.resolve_ips", resolve)


@pytest.fixture
def secrets() -> Secrets:
    return Secrets(api_token=API_TOKEN, webhook_secret=WEBHOOK_SECRET, source="env")


@pytest.fixture
def database(config: Config) -> Database:
    db = Database(config.db_path)
    db.init()
    return db


def write_wav(path: Path, *, seconds: float = 1.0, rate: int = 8000) -> Path:
    """A silent mono PCM WAV, standing in for a telephony voicemail."""
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<h", 0) * frames)
    return path


@pytest.fixture
def make_wav():
    return write_wav


@pytest.fixture
def fake_ffmpeg(tmp_path: Path) -> Path:
    """A stand-in ffmpeg: emits a 16 kHz mono WAV, or fails on a bogus input."""
    script = tmp_path / "fake-ffmpeg"
    script.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import sys, wave, struct

            args = sys.argv[1:]
            source = args[args.index("-i") + 1]
            dest = args[-1]
            with open(source, "rb") as handle:
                head = handle.read(4)
            if head != b"RIFF":
                sys.stderr.write("Invalid data found when processing input\\n")
                sys.exit(1)
            with wave.open(dest, "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(16000)
                out.writeframes(struct.pack("<h", 0) * 16000)
            """
        )
    )
    script.chmod(0o755)
    return script
