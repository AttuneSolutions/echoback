from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import API_TOKEN, write_wav
from echoback.api import create_app
from echoback.config import Config
from echoback.db import Database
from echoback.secrets_store import Secrets

AUTH = {"Authorization": f"Bearer {API_TOKEN}"}
CALLBACK = "https://receiver.test/resume/abc123"


def client_for(config: Config, secrets: Secrets) -> Iterator[TestClient]:
    app = create_app(config, secrets=secrets, run_background=False)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client(config: Config, secrets: Secrets) -> Iterator[TestClient]:
    yield from client_for(config, secrets)


@pytest.fixture
def voicemail(tmp_path: Path) -> bytes:
    return write_wav(tmp_path / "voicemail.wav").read_bytes()


def submit(client: TestClient, voicemail: bytes, **fields):
    data = {"callback_url": CALLBACK}
    data.update({key: value for key, value in fields.items() if value is not None})
    return client.post(
        "/jobs",
        headers=AUTH,
        files={"file": ("voicemail.wav", voicemail, "audio/wav")},
        data=data,
    )


# ---- health -------------------------------------------------------------


def test_health_needs_no_auth(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["queue_depth"] == 0
    assert body["worker"] == "idle"
    assert body["model_default"] == "small"


def test_health_reports_a_dead_worker(client: TestClient) -> None:
    """A worker task that finished must not keep reporting itself as idle."""

    class FinishedTask:
        def get_name(self) -> str:
            return "echoback-worker"

        def done(self) -> bool:
            return True

    client.app.state.tasks = [FinishedTask()]
    body = client.get("/health").json()
    assert body["worker"] == "stopped"
    assert body["status"] == "degraded"


# ---- auth ---------------------------------------------------------------


def test_missing_token_is_rejected(client: TestClient, voicemail: bytes) -> None:
    response = client.post(
        "/jobs",
        files={"file": ("voicemail.wav", voicemail, "audio/wav")},
        data={"callback_url": CALLBACK},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"
    assert response.headers["www-authenticate"] == "Bearer"


def test_wrong_token_is_rejected(client: TestClient, voicemail: bytes) -> None:
    response = client.post(
        "/jobs",
        headers={"Authorization": "Bearer nope"},
        files={"file": ("voicemail.wav", voicemail, "audio/wav")},
        data={"callback_url": CALLBACK},
    )
    assert response.status_code == 401


def test_status_endpoint_requires_auth(client: TestClient) -> None:
    assert client.get("/jobs/whatever").status_code == 401


# ---- submission ---------------------------------------------------------


def test_submit_returns_202_and_queues_the_job(
    client: TestClient, voicemail: bytes, config: Config
) -> None:
    response = submit(client, voicemail, job_ref="vm-2026-08-01-0042")
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["job_ref"] == "vm-2026-08-01-0042"
    assert body["status_url"] == f"/jobs/{body['job_id']}"

    stored = Database(config.db_path).get_job(body["job_id"])
    assert stored is not None
    assert stored.callback_url == CALLBACK
    assert stored.model == "small"
    assert Path(stored.audio_path).exists()
    assert Path(stored.audio_path).read_bytes() == voicemail

    assert client.get("/health").json()["queue_depth"] == 1


def test_job_ref_is_optional(client: TestClient, voicemail: bytes) -> None:
    body = submit(client, voicemail).json()
    assert body["job_ref"] is None


def test_model_override_within_allowlist(client: TestClient, voicemail: bytes) -> None:
    body = submit(client, voicemail, model="medium").json()
    assert body["status"] == "queued"
    assert client.get(f"/jobs/{body['job_id']}", headers=AUTH).json()["model"] == "medium"


def test_unknown_model_is_rejected(client: TestClient, voicemail: bytes) -> None:
    response = submit(client, voicemail, model="giant")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "MODEL_NOT_ALLOWED"


@pytest.mark.parametrize("bad_url", ["", "not-a-url", "ftp://host/path", "/relative/path"])
def test_malformed_callback_url_is_rejected(
    client: TestClient, voicemail: bytes, bad_url: str
) -> None:
    response = client.post(
        "/jobs",
        headers=AUTH,
        files={"file": ("voicemail.wav", voicemail, "audio/wav")},
        data={"callback_url": bad_url},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_CALLBACK_URL"


@pytest.mark.parametrize(
    "blocked_url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://127.0.0.1:8910/inference",  # this container's own whisper-server
        "https://metadata.evil.test/resume",  # DNS pointing at metadata
    ],
)
def test_ssrf_targets_are_rejected(
    client: TestClient, voicemail: bytes, config: Config, blocked_url: str
) -> None:
    response = client.post(
        "/jobs",
        headers=AUTH,
        files={"file": ("voicemail.wav", voicemail, "audio/wav")},
        data={"callback_url": blocked_url},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "CALLBACK_HOST_NOT_ALLOWED"
    assert list(config.audio_dir.iterdir()) == [], "a refused job must not leave audio behind"


def test_private_lan_receiver_is_accepted(client: TestClient, voicemail: bytes) -> None:
    """Activepieces on the same LAN is the expected deployment, not an attack."""
    response = client.post(
        "/jobs",
        headers=AUTH,
        files={"file": ("voicemail.wav", voicemail, "audio/wav")},
        data={"callback_url": "http://10.4.0.9:8080/resume/abc"},
    )
    assert response.status_code == 202


def test_callback_allow_list_is_enforced(
    config: Config, secrets: Secrets, voicemail: bytes
) -> None:
    restricted = dataclasses.replace(config, callback_allowed_hosts=("activepieces.internal",))
    for client in client_for(restricted, secrets):
        assert submit(client, voicemail).status_code == 400
        allowed = client.post(
            "/jobs",
            headers=AUTH,
            files={"file": ("voicemail.wav", voicemail, "audio/wav")},
            data={"callback_url": "http://activepieces.internal/resume"},
        )
        assert allowed.status_code == 202


def test_missing_file_is_rejected(client: TestClient) -> None:
    response = client.post("/jobs", headers=AUTH, data={"callback_url": CALLBACK})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "MISSING_FILE"


def test_empty_file_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/jobs",
        headers=AUTH,
        files={"file": ("empty.wav", b"", "audio/wav")},
        data={"callback_url": CALLBACK},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "EMPTY_FILE"


def test_oversized_upload_returns_413(config: Config, secrets: Secrets) -> None:
    tiny = dataclasses.replace(config, max_upload_mb=1)
    for client in client_for(tiny, secrets):
        response = client.post(
            "/jobs",
            headers=AUTH,
            files={"file": ("big.wav", b"\0" * (2 * 1024 * 1024), "audio/wav")},
            data={"callback_url": CALLBACK},
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"
        assert list(tiny.audio_dir.iterdir()) == [], "rejected upload must not linger on disk"


def test_queue_depth_limit_returns_429(config: Config, secrets: Secrets, voicemail: bytes) -> None:
    shallow = dataclasses.replace(config, max_queue_depth=1)
    for client in client_for(shallow, secrets):
        assert submit(client, voicemail).status_code == 202
        response = submit(client, voicemail)
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "QUEUE_FULL"


def test_vocabulary_hint_is_stored(client: TestClient, voicemail: bytes, config: Config) -> None:
    hint = "invoice, purchase order, back-order; Acme Holdings, A. Patel"
    job_id = submit(client, voicemail, vocabulary_hint=f"  {hint}  ").json()["job_id"]
    assert Database(config.db_path).get_job(job_id).vocabulary_hint == hint


def test_vocabulary_hint_defaults_to_the_configured_value(
    config: Config, secrets: Secrets, voicemail: bytes
) -> None:
    with_default = dataclasses.replace(config, vocabulary_hint="invoice, back-order")
    for client in client_for(with_default, secrets):
        stored = Database(with_default.db_path)
        job_id = submit(client, voicemail).json()["job_id"]
        assert stored.get_job(job_id).vocabulary_hint == "invoice, back-order"
        # An explicitly blank field is the same as omitting it.
        blank_id = submit(client, voicemail, vocabulary_hint="   ").json()["job_id"]
        assert stored.get_job(blank_id).vocabulary_hint == "invoice, back-order"


def test_oversized_vocabulary_hint_is_rejected(client: TestClient, voicemail: bytes) -> None:
    response = submit(client, voicemail, vocabulary_hint="x" * 1001)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_VOCABULARY_HINT"


def test_vocabulary_hint_is_not_echoed_in_status(client: TestClient, voicemail: bytes) -> None:
    job_id = submit(client, voicemail, vocabulary_hint="Acme Holdings").json()["job_id"]
    response = client.get(f"/jobs/{job_id}", headers=AUTH)
    assert "vocabulary_hint" not in response.json()
    assert "Acme" not in response.text


def test_long_job_ref_is_rejected(client: TestClient, voicemail: bytes) -> None:
    response = submit(client, voicemail, job_ref="x" * 300)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_JOB_REF"


@pytest.mark.parametrize("forged", ["ref\ninjected log line", "ref\r\nINFO fake", "ref\x00null"])
def test_control_characters_in_job_ref_are_rejected(
    client: TestClient, voicemail: bytes, forged: str
) -> None:
    """job_ref is logged, so a newline in it would let a caller forge log lines."""
    response = submit(client, voicemail, job_ref=forged)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_JOB_REF"


def test_control_characters_in_vocabulary_hint_are_rejected(
    client: TestClient, voicemail: bytes
) -> None:
    response = submit(client, voicemail, vocabulary_hint="Aroha\x07Ngatai")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_VOCABULARY_HINT"


def test_queued_bytes_limit_returns_429(config: Config, secrets: Secrets, voicemail: bytes) -> None:
    """Depth alone is not backpressure — 1000 × 25 MB would be tens of gigabytes."""
    byte_capped = dataclasses.replace(config, max_queue_mb=0)
    for client in client_for(byte_capped, secrets):
        response = submit(client, voicemail)
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "QUEUE_FULL"
        assert "MAX_QUEUE_MB" in response.json()["error"]["message"]


def test_upload_size_is_recorded_for_backpressure(
    client: TestClient, voicemail: bytes, config: Config
) -> None:
    job_id = submit(client, voicemail).json()["job_id"]
    database = Database(config.db_path)
    assert database.get_job(job_id).audio_bytes == len(voicemail)
    assert database.queued_bytes() == len(voicemail)


def test_status_url_is_relative_without_host_url(client: TestClient, voicemail: bytes) -> None:
    body = submit(client, voicemail).json()
    assert body["status_url"] == f"/jobs/{body['job_id']}"


def test_status_url_is_absolute_with_host_url(
    config: Config, secrets: Secrets, voicemail: bytes
) -> None:
    hosted = dataclasses.replace(config, host_url="https://voicemail.example.com")
    for client in client_for(hosted, secrets):
        body = submit(client, voicemail).json()
        assert body["status_url"] == f"https://voicemail.example.com/jobs/{body['job_id']}"


# ---- status ------------------------------------------------------------


def test_status_shape_for_queued_job(client: TestClient, voicemail: bytes) -> None:
    job_id = submit(client, voicemail, job_ref="ref-1").json()["job_id"]
    body = client.get(f"/jobs/{job_id}", headers=AUTH).json()
    assert set(body) == {
        "job_id",
        "job_ref",
        "status",
        "text",
        "model",
        "duration_ms",
        "created_at",
        "completed_at",
        "delivered_at",
        "error",
    }
    assert body["delivered_at"] is None
    assert body["status"] == "queued"
    assert body["text"] is None
    assert body["error"] is None
    assert body["created_at"].endswith("Z")


def test_status_reports_completed_transcript(
    client: TestClient, voicemail: bytes, config: Config
) -> None:
    database = Database(config.db_path)
    job_id = submit(client, voicemail).json()["job_id"]
    database.mark_transcribed(job_id, text="Hi, it's Aroha", duration_ms=30120)

    # Awaiting the webhook ack: the transcript is already fetchable here.
    body = client.get(f"/jobs/{job_id}", headers=AUTH).json()
    assert body["status"] == "transcribed"
    assert body["text"] == "Hi, it's Aroha"
    assert body["duration_ms"] == 30120
    assert body["completed_at"] is not None

    database.mark_delivered(job_id, attempts=1)
    body = client.get(f"/jobs/{job_id}", headers=AUTH).json()
    assert body["status"] == "done"
    assert body["text"] == "Hi, it's Aroha"


def test_status_keeps_the_transcript_after_callbacks_are_exhausted(
    client: TestClient, voicemail: bytes, config: Config
) -> None:
    database = Database(config.db_path)
    job_id = submit(client, voicemail).json()["job_id"]
    database.mark_transcribed(job_id, text="Hi, it's Aroha", duration_ms=30120)
    database.mark_callback_failed(job_id, attempts=5)

    body = client.get(f"/jobs/{job_id}", headers=AUTH).json()
    assert body["status"] == "callback_failed"
    assert body["text"] == "Hi, it's Aroha", "the transcript must stay fetchable"


def test_status_reports_failure_error(client: TestClient, voicemail: bytes, config: Config) -> None:
    job_id = submit(client, voicemail).json()["job_id"]
    Database(config.db_path).mark_failed(
        job_id, code="AUDIO_DECODE_FAILED", message="ffmpeg could not decode the uploaded file"
    )
    body = client.get(f"/jobs/{job_id}", headers=AUTH).json()
    assert body["status"] == "failed"
    assert body["text"] is None
    assert body["error"] == {
        "code": "AUDIO_DECODE_FAILED",
        "message": "ffmpeg could not decode the uploaded file",
    }


def test_unknown_job_returns_404(client: TestClient) -> None:
    response = client.get("/jobs/00000000-0000-0000-0000-000000000000", headers=AUTH)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


# ---- startup ------------------------------------------------------------


def test_startup_requeues_stranded_processing_jobs(config: Config, secrets: Secrets) -> None:
    database = Database(config.db_path)
    database.init()
    database.insert_job(
        job_id="stranded",
        job_ref=None,
        callback_url=CALLBACK,
        model="small",
        audio_path=str(config.audio_dir / "stranded.wav"),
    )
    database.claim_next_queued()
    assert database.get_job("stranded").status == "processing"

    for client in client_for(config, secrets):
        body = client.get("/jobs/stranded", headers=AUTH).json()
        assert body["status"] == "queued"
