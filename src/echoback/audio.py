"""Audio normalisation via the ffmpeg CLI.

ffmpeg is invoked as an external subprocess, never linked as a library, so its
LGPL/GPL terms stay clear of this project's MIT source (see THIRD_PARTY_LICENSES).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import wave
from pathlib import Path

log = logging.getLogger("echoback.audio")

TARGET_RATE = 16000
TARGET_CHANNELS = 1


class AudioDecodeError(Exception):
    """ffmpeg could not decode the uploaded file."""


async def normalize(
    source: Path,
    dest: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    timeout: float = 120.0,
) -> Path:
    """Transcode any ffmpeg-decodable input to 16 kHz mono PCM WAV."""
    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ar",
        str(TARGET_RATE),
        "-ac",
        str(TARGET_CHANNELS),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(dest),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:  # ffmpeg missing from the image
        raise AudioDecodeError(f"ffmpeg binary {ffmpeg_bin!r} not found") from exc

    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        raise AudioDecodeError(f"ffmpeg timed out after {timeout:.0f}s") from exc

    if proc.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        detail = (stderr or b"").decode("utf-8", "replace").strip().splitlines()
        tail = detail[-1] if detail else f"exit code {proc.returncode}"
        raise AudioDecodeError(f"ffmpeg could not decode the uploaded file: {tail}")
    return dest


def duration_ms(wav_path: Path) -> int | None:
    """Duration of a PCM WAV, read from its header. None if unreadable."""
    try:
        with contextlib.closing(wave.open(str(wav_path), "rb")) as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
        if not rate:
            return None
        return round(frames / rate * 1000)
    except (OSError, EOFError, wave.Error):
        log.debug("could not read duration from %s", wav_path)
        return None
