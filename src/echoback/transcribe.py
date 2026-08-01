"""Offline batch transcription over local files, using the service's own pipeline.

    python -m echoback.transcribe [--model small] [--json] FILE...

Same ffmpeg normalisation and whisper.cpp engine the worker uses, minus the queue
and the webhook — for eyeballing transcript quality on real recordings.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

from . import audio as audio_mod
from .config import Config, load_config
from .whisper import TranscriptionError, WhisperEngine


@dataclasses.dataclass
class Result:
    path: Path
    model: str
    duration_ms: int | None = None
    elapsed_ms: int | None = None
    text: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "file": self.path.name,
            "model": self.model,
            "duration_ms": self.duration_ms,
            "elapsed_ms": self.elapsed_ms,
            "text": self.text,
            "error": self.error,
        }


async def transcribe_file(engine: WhisperEngine, config: Config, path: Path, model: str) -> Result:
    result = Result(path=path, model=model)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="echoback-transcribe-") as work:
        normalized = Path(work) / "normalized.wav"
        try:
            await audio_mod.normalize(path, normalized, ffmpeg_bin=config.ffmpeg_bin)
        except audio_mod.AudioDecodeError as exc:
            result.error = f"AUDIO_DECODE_FAILED: {exc}"
            return result
        result.duration_ms = audio_mod.duration_ms(normalized)
        try:
            result.text = await engine.transcribe(normalized, model)
        except TranscriptionError as exc:
            result.error = f"{exc.code}: {exc}"
            return result
    result.elapsed_ms = round((time.monotonic() - started) * 1000)
    return result


async def transcribe_paths(paths: list[Path], *, model: str | None = None) -> list[Result]:
    config = load_config()
    resolved_model = model or config.model_default
    if not config.is_model_allowed(resolved_model):
        raise SystemExit(f"model {resolved_model!r} is not in {list(config.model_allowlist)}")
    engine = WhisperEngine(config)
    try:
        return [await transcribe_file(engine, config, path, resolved_model) for path in paths]
    finally:
        await engine.stop()


def format_table(results: list[Result]) -> str:
    lines = []
    for result in results:
        header = result.path.name
        if result.duration_ms is not None:
            header += f"  ({result.duration_ms / 1000:.1f}s audio"
            if result.elapsed_ms is not None:
                header += f", {result.elapsed_ms / 1000:.1f}s to transcribe"
            header += ")"
        lines.append(header)
        lines.append("-" * len(header))
        lines.append(result.error or (result.text or "<empty transcript>"))
        lines.append("")
    return "\n".join(lines)


SKIP_SUFFIXES = {".md", ".txt", ".json", ".gitkeep", ""}


def expand(paths: list[Path]) -> list[Path]:
    """Files as given; a directory contributes the audio-looking files inside it."""
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(
                sorted(
                    child
                    for child in path.iterdir()
                    if child.is_file() and child.suffix.lower() not in SKIP_SUFFIXES
                )
            )
        else:
            expanded.append(path)
    return expanded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="echoback.transcribe", description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="audio files, or directories of them")
    parser.add_argument("--model", help="model name (defaults to MODEL_DEFAULT)")
    parser.add_argument("--json", action="store_true", help="emit JSON lines instead of a table")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    paths = expand(args.paths)
    if not paths:
        parser.error("no audio files found in the given paths")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        parser.error(f"not a file: {', '.join(str(path) for path in missing)}")

    results = asyncio.run(transcribe_paths(paths, model=args.model))
    if args.json:
        for result in results:
            print(json.dumps(result.as_dict(), ensure_ascii=False))
    else:
        print(format_table(results))
    return 1 if any(result.error for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
