"""whisper.cpp transcription engine.

The default model runs in a **resident** ``whisper-server`` child process so the
weights are loaded once and stay warm — model load dominates runtime for a
30-second voicemail. A job that overrides the model to something other than the
default is transcribed with a one-shot ``whisper-cli`` invocation instead, which
keeps memory bounded to a single resident model.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

import httpx

from .config import Config

log = logging.getLogger("echoback.whisper")


class TranscriptionError(Exception):
    def __init__(self, message: str, *, code: str = "TRANSCRIPTION_FAILED") -> None:
        super().__init__(message)
        self.code = code


class WhisperEngine:
    """Owns the resident whisper.cpp server and the one-shot CLI fallback."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._base_url = f"http://{config.whisper_host}:{config.whisper_port}"

    # ---- lifecycle ------------------------------------------------------

    @property
    def resident_model(self) -> str:
        return self._config.model_default

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        async with self._lock:
            await self._ensure_server_locked()

    async def stop(self) -> None:
        async with self._lock:
            proc, self._proc = self._proc, None
            if proc is None or proc.returncode is not None:
                return
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
            log.info("whisper-server stopped")

    async def _ensure_server_locked(self) -> None:
        if self.is_running():
            return
        if self._proc is not None:
            log.warning("whisper-server exited with code %s — restarting", self._proc.returncode)
            self._proc = None

        model_path = self._require_model(self.resident_model)
        cmd = [
            self._config.whisper_server_bin,
            "--model",
            str(model_path),
            "--host",
            self._config.whisper_host,
            "--port",
            str(self._config.whisper_port),
            "--no-timestamps",
        ]
        if self._config.whisper_threads > 0:
            cmd += ["--threads", str(self._config.whisper_threads)]

        log.info("starting whisper-server with model %s", self.resident_model)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise TranscriptionError(
                f"whisper-server binary {self._config.whisper_server_bin!r} not found",
                code="ENGINE_UNAVAILABLE",
            ) from exc

        await self._wait_until_ready()

    async def _wait_until_ready(self) -> None:
        deadline = asyncio.get_running_loop().time() + self._config.whisper_startup_timeout
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                if self._proc is None or self._proc.returncode is not None:
                    raise TranscriptionError(
                        "whisper-server exited during startup", code="ENGINE_UNAVAILABLE"
                    )
                with contextlib.suppress(httpx.HTTPError):
                    await client.get(self._base_url + "/")
                    log.info("whisper-server ready on %s", self._base_url)
                    return
                if asyncio.get_running_loop().time() >= deadline:
                    raise TranscriptionError(
                        f"whisper-server did not become ready within "
                        f"{self._config.whisper_startup_timeout:.0f}s",
                        code="ENGINE_UNAVAILABLE",
                    )
                await asyncio.sleep(0.5)

    # ---- transcription --------------------------------------------------

    async def transcribe(
        self, wav_path: Path, model: str, vocabulary_hint: str | None = None
    ) -> str:
        """Transcribe ``wav_path``.

        ``vocabulary_hint`` is passed to whisper.cpp as the initial prompt: domain
        phrasing and candidate proper nouns that bias decoding without any training.
        """
        if model == self.resident_model:
            async with self._lock:
                await self._ensure_server_locked()
            return await self._transcribe_resident(wav_path, vocabulary_hint)
        return await self._transcribe_cli(wav_path, model, vocabulary_hint)

    async def _transcribe_resident(self, wav_path: Path, vocabulary_hint: str | None) -> str:
        data = {
            "temperature": "0.0",
            "response_format": "json",
            "no_timestamps": "true",
        }
        if vocabulary_hint:
            data["prompt"] = vocabulary_hint
        try:
            async with httpx.AsyncClient(timeout=self._config.whisper_request_timeout) as client:
                with wav_path.open("rb") as handle:
                    response = await client.post(
                        self._base_url + "/inference",
                        files={"file": (wav_path.name, handle, "audio/wav")},
                        data=data,
                    )
        except httpx.HTTPError as exc:
            raise TranscriptionError(f"whisper-server request failed: {exc}") from exc

        if response.status_code >= 400:
            raise TranscriptionError(f"whisper-server returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise TranscriptionError("whisper-server returned a non-JSON body") from exc
        text = payload.get("text")
        if not isinstance(text, str):
            raise TranscriptionError("whisper-server response had no text field")
        return text.strip()

    async def _transcribe_cli(self, wav_path: Path, model: str, vocabulary_hint: str | None) -> str:
        model_path = self._require_model(model)
        cmd = [
            self._config.whisper_cli_bin,
            "--model",
            str(model_path),
            "--file",
            str(wav_path),
            "--no-timestamps",
            "--no-prints",
        ]
        if vocabulary_hint:
            cmd += ["--prompt", vocabulary_hint]
        if self._config.whisper_threads > 0:
            cmd += ["--threads", str(self._config.whisper_threads)]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise TranscriptionError(
                f"whisper-cli binary {self._config.whisper_cli_bin!r} not found",
                code="ENGINE_UNAVAILABLE",
            ) from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._config.whisper_request_timeout
            )
        except asyncio.TimeoutError as exc:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            raise TranscriptionError("whisper-cli timed out") from exc

        if proc.returncode != 0:
            detail = (stderr or b"").decode("utf-8", "replace").strip().splitlines()
            tail = detail[-1] if detail else f"exit code {proc.returncode}"
            raise TranscriptionError(f"whisper-cli failed: {tail}")
        return stdout.decode("utf-8", "replace").strip()

    def _require_model(self, model: str) -> Path:
        path = self._config.model_path(model)
        if not path.exists():
            raise TranscriptionError(
                f"model file for {model!r} is not present in the image at {path}",
                code="MODEL_UNAVAILABLE",
            )
        return path
