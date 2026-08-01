from __future__ import annotations

import dataclasses
import socket
import textwrap
from pathlib import Path

import pytest

from echoback.config import Config
from echoback.whisper import TranscriptionError, WhisperEngine


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def fake_server(tmp_path: Path) -> Path:
    """A stand-in whisper-server: answers `/` and returns JSON from `/inference`."""
    script = tmp_path / "fake-whisper-server"
    script.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json, sys
            from http.server import BaseHTTPRequestHandler, HTTPServer

            args = sys.argv[1:]
            port = int(args[args.index("--port") + 1])
            model = args[args.index("--model") + 1]

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *a):
                    pass

                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"whisper.cpp")

                def do_POST(self):
                    length = int(self.headers.get("content-length", 0))
                    body = self.rfile.read(length)
                    # The engine sends the hint as a multipart form field; look for it in
                    # the raw body rather than parsing multipart here.
                    prompt = "yes" if b"hint marker" in body else "no"
                    payload = json.dumps(
                        {
                            "text": "  transcribed %d bytes with %s prompt=%s  "
                            % (len(body), model, prompt)
                        }
                    ).encode()
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

            HTTPServer(("127.0.0.1", port), Handler).serve_forever()
            """
        )
    )
    script.chmod(0o755)
    return script


@pytest.fixture
def fake_cli(tmp_path: Path) -> Path:
    """A stand-in whisper-cli: prints the transcript on stdout."""
    script = tmp_path / "fake-whisper-cli"
    script.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import sys
            args = sys.argv[1:]
            model = args[args.index("--model") + 1]
            prompt = args[args.index("--prompt") + 1] if "--prompt" in args else "none"
            sys.stderr.write("whisper log noise\\n")
            print("cli transcript from %s prompt=%s" % (model, prompt))
            """
        )
    )
    script.chmod(0o755)
    return script


def engine_config(config: Config, **overrides) -> Config:
    return dataclasses.replace(
        config,
        whisper_port=free_port(),
        whisper_startup_timeout=20.0,
        **overrides,
    )


async def test_resident_server_transcribes_default_model(
    config: Config, fake_server: Path, fake_cli: Path, make_wav, tmp_path: Path
) -> None:
    engine = WhisperEngine(
        engine_config(config, whisper_server_bin=str(fake_server), whisper_cli_bin=str(fake_cli))
    )
    wav = make_wav(tmp_path / "normalized.wav")
    try:
        await engine.start()
        assert engine.is_running()
        text = await engine.transcribe(wav, "small")
        assert text.startswith("transcribed")
        assert "ggml-small.bin" in text
        assert text == text.strip()
    finally:
        await engine.stop()
    assert not engine.is_running()


async def test_non_default_model_uses_the_cli(
    config: Config, fake_server: Path, fake_cli: Path, make_wav, tmp_path: Path
) -> None:
    engine = WhisperEngine(
        engine_config(config, whisper_server_bin=str(fake_server), whisper_cli_bin=str(fake_cli))
    )
    wav = make_wav(tmp_path / "normalized.wav")
    text = await engine.transcribe(wav, "medium")
    assert "cli transcript" in text
    assert "ggml-medium.bin" in text
    assert not engine.is_running(), "the CLI path must not start the resident server"


async def test_vocabulary_hint_reaches_the_resident_server(
    config: Config, fake_server: Path, fake_cli: Path, make_wav, tmp_path: Path
) -> None:
    engine = WhisperEngine(
        engine_config(config, whisper_server_bin=str(fake_server), whisper_cli_bin=str(fake_cli))
    )
    wav = make_wav(tmp_path / "normalized.wav")
    try:
        await engine.start()
        assert "prompt=no" in await engine.transcribe(wav, "small")
        assert "prompt=yes" in await engine.transcribe(wav, "small", "hint marker, purchase order")
    finally:
        await engine.stop()


async def test_vocabulary_hint_reaches_the_cli(
    config: Config, fake_server: Path, fake_cli: Path, make_wav, tmp_path: Path
) -> None:
    engine = WhisperEngine(
        engine_config(config, whisper_server_bin=str(fake_server), whisper_cli_bin=str(fake_cli))
    )
    wav = make_wav(tmp_path / "normalized.wav")
    assert "prompt=none" in await engine.transcribe(wav, "medium")
    assert "prompt=Acme Holdings" in await engine.transcribe(wav, "medium", "Acme Holdings")


async def test_missing_model_file_is_reported(
    config: Config, fake_server: Path, fake_cli: Path, make_wav, tmp_path: Path
) -> None:
    engine = WhisperEngine(
        engine_config(config, whisper_server_bin=str(fake_server), whisper_cli_bin=str(fake_cli))
    )
    (config.model_dir / "ggml-base.bin").unlink(missing_ok=True)
    with pytest.raises(TranscriptionError) as excinfo:
        await engine.transcribe(make_wav(tmp_path / "n.wav"), "base")
    assert excinfo.value.code == "MODEL_UNAVAILABLE"


async def test_missing_server_binary_is_reported(config: Config) -> None:
    engine = WhisperEngine(
        engine_config(config, whisper_server_bin=str(config.data_dir / "no-such-binary"))
    )
    with pytest.raises(TranscriptionError) as excinfo:
        await engine.start()
    assert excinfo.value.code == "ENGINE_UNAVAILABLE"


async def test_failing_cli_reports_transcription_failure(
    config: Config, tmp_path: Path, make_wav
) -> None:
    broken = tmp_path / "broken-cli"
    broken.write_text("#!/bin/sh\necho 'whisper: fatal' >&2\nexit 3\n")
    broken.chmod(0o755)
    engine = WhisperEngine(engine_config(config, whisper_cli_bin=str(broken)))
    with pytest.raises(TranscriptionError) as excinfo:
        await engine.transcribe(make_wav(tmp_path / "n.wav"), "medium")
    assert excinfo.value.code == "TRANSCRIPTION_FAILED"
    assert "fatal" in str(excinfo.value)


async def test_server_that_exits_immediately_fails_startup(config: Config, tmp_path: Path) -> None:
    quitter = tmp_path / "quitting-server"
    quitter.write_text("#!/bin/sh\nexit 1\n")
    quitter.chmod(0o755)
    engine = WhisperEngine(engine_config(config, whisper_server_bin=str(quitter)))
    with pytest.raises(TranscriptionError) as excinfo:
        await engine.start()
    assert excinfo.value.code == "ENGINE_UNAVAILABLE"
