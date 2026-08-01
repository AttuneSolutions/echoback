"""Real-recording smoke test.

Drop actual voicemails (any format ffmpeg can decode) into `tests/samples/` — the
directory is gitignored, so nothing personal is ever committed — and this module
runs each one through the real ffmpeg + whisper.cpp pipeline and prints the
transcript. It skips itself when there are no samples or when the binaries and
model weights are not available on this machine, so CI and a bare checkout stay
green.

    pytest tests/test_samples.py -s              # transcripts go to stdout
    ECHOBACK_SAMPLE_MODEL=medium pytest tests/test_samples.py -s

Requires `ffmpeg`, `whisper-server`/`whisper-cli` and `MODEL_DIR` on PATH/env. The
easiest way to get those is the built image — see scripts/transcribe-samples.sh.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from echoback.config import load_config
from echoback.transcribe import format_table, transcribe_paths

SAMPLES_DIR = Path(__file__).parent / "samples"
IGNORED_SUFFIXES = {".md", ".txt", ".gitkeep", ""}


def sample_files() -> list[Path]:
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(
        path
        for path in SAMPLES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() not in IGNORED_SUFFIXES
    )


def missing_prerequisite() -> str | None:
    """Why this machine cannot run the real pipeline, or None if it can."""
    config = load_config()
    model = os.environ.get("ECHOBACK_SAMPLE_MODEL", config.model_default)
    for binary in (config.ffmpeg_bin, config.whisper_server_bin, config.whisper_cli_bin):
        if shutil.which(binary) is None and not Path(binary).is_file():
            return f"{binary} is not installed"
    if not config.model_path(model).is_file():
        return f"model weights for {model!r} are not at {config.model_path(model)}"
    return None


SAMPLES = sample_files()

pytestmark = [
    pytest.mark.skipif(
        not SAMPLES,
        reason=f"no sample recordings in {SAMPLES_DIR} — drop some voicemails in to enable",
    ),
]


@pytest.fixture(scope="module")
def prerequisites() -> None:
    reason = missing_prerequisite()
    if reason:
        pytest.skip(f"real engine unavailable: {reason}")


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda path: path.name)
def test_sample_transcribes(sample: Path, prerequisites: None, capsys) -> None:
    import asyncio

    model = os.environ.get("ECHOBACK_SAMPLE_MODEL") or None
    results = asyncio.run(transcribe_paths([sample], model=model))
    result = results[0]

    with capsys.disabled():
        print()
        print(format_table(results), end="")

    assert result.error is None, result.error
    assert result.text, "transcript came back empty — check the recording and the model"
    assert result.duration_ms and result.duration_ms > 0
