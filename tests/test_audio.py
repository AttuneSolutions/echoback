from __future__ import annotations

import wave
from pathlib import Path

import pytest

from echoback import audio


async def test_normalize_produces_16k_mono_wav(tmp_path: Path, fake_ffmpeg: Path, make_wav) -> None:
    source = make_wav(tmp_path / "voicemail.wav", seconds=1.0, rate=8000)
    dest = tmp_path / "normalized.wav"

    await audio.normalize(source, dest, ffmpeg_bin=str(fake_ffmpeg))

    assert dest.exists()
    with wave.open(str(dest), "rb") as handle:
        assert handle.getframerate() == audio.TARGET_RATE
        assert handle.getnchannels() == audio.TARGET_CHANNELS


async def test_normalize_raises_on_undecodable_input(tmp_path: Path, fake_ffmpeg: Path) -> None:
    source = tmp_path / "corrupt.wav"
    source.write_bytes(b"this is not audio at all")
    dest = tmp_path / "normalized.wav"

    with pytest.raises(audio.AudioDecodeError, match="could not decode"):
        await audio.normalize(source, dest, ffmpeg_bin=str(fake_ffmpeg))


async def test_normalize_raises_when_ffmpeg_missing(tmp_path: Path, make_wav) -> None:
    source = make_wav(tmp_path / "voicemail.wav")
    with pytest.raises(audio.AudioDecodeError, match="not found"):
        await audio.normalize(
            source, tmp_path / "out.wav", ffmpeg_bin=str(tmp_path / "no-such-ffmpeg")
        )


def test_duration_ms_from_wav_header(tmp_path: Path, make_wav) -> None:
    source = make_wav(tmp_path / "two-seconds.wav", seconds=2.0, rate=16000)
    assert audio.duration_ms(source) == 2000


def test_duration_ms_returns_none_for_non_wav(tmp_path: Path) -> None:
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"nope")
    assert audio.duration_ms(junk) is None
    assert audio.duration_ms(tmp_path / "missing.wav") is None
