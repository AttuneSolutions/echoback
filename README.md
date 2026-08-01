# Echoback

Offline, self-hosted voicemail-to-text for PBX systems. Send a voicemail file,
get the transcript delivered to your webhook. Powered by `whisper.cpp`. No cloud,
no GPU, MIT-licensed.

_It listens to the message, and calls your automation back with the text._

**Why:** PBX systems like 3CX and Yeastar leave voicemails as audio files. This
service transcribes them locally and hands the text to your automation platform
(built for [Activepieces](https://www.activepieces.com/)' wait-for-webhook, works
with anything).

## Features

- Fully offline transcription (`whisper.cpp`, CPU-only).
- Single small container; SQLite queue; one job at a time.
- Multipart upload in, signed webhook out.
- Handles telephony WAV (PCM/ADPCM) and MP3 via ffmpeg normalisation.
- Multilingual model by default — good with te reo Māori names mixed into English.
- Auto-generated secrets on first boot; audio deleted after transcription; 1-hour
  retention.

## Quick start

```bash
docker run -d --name echoback \
  -p 8080:80 -v echoback_data:/data \
  ghcr.io/attunesolutions/echoback:latest
docker logs echoback   # grab the one-time secrets
```

The first boot prints both secrets once:

```
================ SAVE THESE — SHOWN ONCE ================
API_TOKEN:      k7Qp....         (use as: Authorization: Bearer <token>)
WEBHOOK_SECRET: 9fA2....         (verify X-Signature with this)
========================================================
```

Front it with a reverse proxy for HTTPS, then POST a voicemail:

```bash
curl -X POST https://voicemail.example.com/jobs \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "file=@voicemail.wav" \
  -F "callback_url=https://your-activepieces/resume/abc123"
```

```json
{
  "job_id": "b3f1c2a7-9d4e-4f8a-8c11-2e6d5a0b1234",
  "job_ref": null,
  "status": "queued",
  "status_url": "/jobs/b3f1c2a7-9d4e-4f8a-8c11-2e6d5a0b1234"
}
```

The transcript is POSTed to your `callback_url` when ready.

## API

All endpoints require `Authorization: Bearer <API_TOKEN>` except `GET /health`.

### `POST /jobs` — submit a voicemail

`multipart/form-data`:

| Field          | Required | Description                                                            |
| -------------- | -------- | ---------------------------------------------------------------------- |
| `file`         | yes      | Voicemail audio, any codec ffmpeg can decode. Max `MAX_UPLOAD_MB` (25). |
| `callback_url` | yes      | Absolute http(s) URL the result is POSTed to. Must pass the address policy (below). |
| `model`        | no       | Model override from `MODEL_ALLOWLIST`. Defaults to `MODEL_DEFAULT`.     |
| `job_ref`      | no       | Your correlation id, echoed back verbatim. Max 256 characters.          |
| `vocabulary_hint` | no    | Words the transcriber should expect (see below). Max 1000 characters. Defaults to `VOCABULARY_HINT`. |

Responses: `202` queued · `400` bad field / unknown model / malformed URL ·
`401` bad token · `413` file over the cap · `429` queue at `MAX_QUEUE_DEPTH`.

Errors are `{"error": {"code": "...", "message": "..."}}`. Codes:
`MISSING_FILE`, `EMPTY_FILE`, `INVALID_CALLBACK_URL`, `CALLBACK_HOST_NOT_ALLOWED`,
`MODEL_NOT_ALLOWED`, `INVALID_JOB_REF`, `INVALID_VOCABULARY_HINT`, `UNAUTHORIZED`,
`PAYLOAD_TOO_LARGE`, `QUEUE_FULL`.

`status_url` is a path by default and an absolute URL when `HOST_URL` is set.

#### Which `callback_url`s are accepted

The service POSTs to whatever URL you give it, so an unrestricted `callback_url` is
a server-side request forgery primitive for anyone holding the API token. The
policy:

- **Allowed:** any public address, and **private LAN ranges** (`10.x`, `192.168.x`,
  `172.16–31.x`) — a self-hosted Activepieces next to this service is the expected
  receiver.
- **Refused** (`400 CALLBACK_HOST_NOT_ALLOWED`): loopback (`127.0.0.0/8`, `::1`),
  link-local (`169.254.0.0/16`, `fe80::/10` — this is what blocks the cloud
  metadata endpoint), unspecified, multicast, and reserved addresses. URLs with
  embedded credentials are refused too.
- **Hostnames are resolved** and *every* address they resolve to must pass, so a DNS
  name pointing at a blocked address is refused. A host that cannot be resolved is
  refused rather than accepted on faith.
- **`CALLBACK_ALLOWED_HOSTS`** narrows this to a named set. A leading dot matches
  subdomains: `.example.com` admits `flows.example.com`. The address policy still
  applies on top — allow-listing a host does not license it to point at loopback.
- **`HOST_URL`**, when set, also refuses a `callback_url` aimed back at this service.

Resolution happens at submission. A host that changes its DNS answer afterwards
(rebinding) is not re-checked at delivery time — the bearer token remains the
primary trust boundary.

#### What to put in `vocabulary_hint`

The hint is passed to whisper as the initial prompt. It biases decoding towards
words it contains — no training, no model changes, and no guarantee. Two things
belong in it:

1. **Domain vocabulary** the caller is likely to use — the terms, product names,
   and jargon your organisation says out loud but a general model spells wrong.
2. **A handful of proper nouns you already expect for this specific call**, such
   as the one or two account or contact names you resolved from the caller ID
   before submitting the job.

Send a shortlist, **not your whole database**. Whisper truncates the prompt at
about 224 tokens (≈900 characters), so a long list loses its tail — and a pile of
unrelated names dilutes the bias and invites the model to hallucinate one that was
never said. A useful hint reads like a fragment of domain speech, not a data dump.

```
invoice, purchase order, back-order, RMA, dispatch; Acme Holdings, A. Patel
```

The hint is stored on the job row for the retention window and handed to the
engine. It is **never** returned by `GET /jobs/{id}`, included in the webhook, or
written to logs: it typically carries personal data, and echoing it back would
widen exposure for no benefit.

### `GET /jobs/{job_id}` — status / result

A debug and reconciliation fallback; the webhook is the primary delivery path.
Works only inside the retention window — afterwards the row is purged and this
returns `404`.

```json
{
  "job_id": "b3f1c2a7-...",
  "job_ref": "vm-2026-08-01-0042",
  "status": "done",
  "text": "Hi, it's Aroha calling from ...",
  "model": "small",
  "duration_ms": 30120,
  "created_at": "2026-08-01T09:15:03Z",
  "completed_at": "2026-08-01T09:15:11Z",
  "delivered_at": "2026-08-01T09:15:12Z",
  "error": null
}
```

`completed_at` is when transcription finished; `delivered_at` is when your receiver
acknowledged the webhook, and is `null` until then.

`status` is one of:

| Status            | Meaning                                                                    |
| ----------------- | -------------------------------------------------------------------------- |
| `queued`          | Accepted, waiting for the worker.                                          |
| `processing`      | Being normalised and transcribed.                                          |
| `transcribed`     | Transcript ready and returned here, but the webhook is not yet acked.      |
| `done`            | Transcript delivered — your receiver answered 2xx.                         |
| `failed`          | Transcription failed; `error` is populated.                                |
| `callback_failed` | Result exists, delivery gave up after `WEBHOOK_ATTEMPTS`.                  |

`text` is populated for `transcribed`, `done`, and `callback_failed`. If you poll
this endpoint rather than waiting for the webhook, treat `transcribed` the same as
`done`.

### `GET /health` — liveness

No auth. Suitable as a container health check.

```json
{
  "status": "ok",
  "queue_depth": 0,
  "worker": "idle",
  "model_default": "small",
  "engine": "ready",
  "version": "1.0.0"
}
```

`worker` is `idle`, `busy`, or `stopped`; `stopped` means the background loop is no
longer running and `status` reports `degraded`. `engine` is `ready` or `down`.

`engine` is `ready` once the resident `whisper-server` is up, `down` if it failed
to start — jobs processed while it is `down` fail with `ENGINE_UNAVAILABLE`.

## Webhook delivery

On a terminal transcription state the service POSTs JSON to `callback_url` with
headers `X-Signature: sha256=<hex>`, `X-Job-Id` and `User-Agent: echoback/<version>`.

Success:

```json
{
  "job_id": "b3f1c2a7-...",
  "job_ref": "vm-2026-08-01-0042",
  "status": "done",
  "text": "Hi, it's Aroha calling from ...",
  "model": "small",
  "duration_ms": 30120,
  "completed_at": "2026-08-01T09:15:11Z",
  "error": null
}
```

Failure — always sent, so a waiting flow branches instead of hanging:

```json
{
  "job_id": "b3f1c2a7-...",
  "job_ref": "vm-2026-08-01-0042",
  "status": "failed",
  "text": null,
  "model": "small",
  "duration_ms": null,
  "completed_at": "2026-08-01T09:15:07Z",
  "error": {
    "code": "AUDIO_DECODE_FAILED",
    "message": "ffmpeg could not decode the uploaded file"
  }
}
```

Failure codes: `AUDIO_DECODE_FAILED`, `AUDIO_MISSING`, `MODEL_UNAVAILABLE`,
`ENGINE_UNAVAILABLE`, `TRANSCRIPTION_FAILED`.

**Delivery is acknowledged.** Only a 2xx from your receiver counts as delivered —
a job becomes `done` when you confirm receipt, not when the transcript is produced.
Redirects are not followed, so a 3xx is treated as a failed attempt, as is any
4xx/5xx, connection error, or timeout.

**Retries.** A failed attempt is retried up to `WEBHOOK_ATTEMPTS` times — 5 by
default — with the delay doubling from `WEBHOOK_BACKOFF`: 2s, 4s, 8s, 16s between
attempts, so five attempts span about a minute. After the last one the job becomes
`callback_failed`; the transcript stays fetchable from `GET /jobs/{job_id}` until
the retention window closes.

**Duplicates.** If the container restarts between your 2xx and the ack being
recorded, the same result is delivered again. Deduplicate on `job_id`.

### Verifying the signature

```python
import hashlib, hmac


def valid(secret: str, raw_body: bytes, header: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header or "")
```

Sign over the **raw request body**, not a re-serialised copy.

## Activepieces integration

1. Add a step that generates a resume URL and pauses the flow (webhook waitpoint).
2. In the preceding HTTP step, POST the voicemail to `/jobs`, passing the resume URL
   as `callback_url`.
3. The flow resumes with `body.text` as the transcript. Branch on
   `body.status == "done"`.

> A known upstream quirk: calling a resume URL after a piece has already failed can
> let the flow continue past the failed step. Echoback always sends an explicit
> `status` (and an `error` object on failure), so branch on `body.status == "done"`
> and treat everything else as the failure path.

## Configuration

| Variable            | Default   | Description                                                  |
| ------------------- | --------- | ------------------------------------------------------------ |
| `PORT`              | `80`      | HTTP listen port. TLS is expected to be terminated upstream. |
| `HOST_URL`          | unset     | Public URL of this service, as fronted by your proxy (e.g. `https://voicemail.example.com`). Makes `status_url` absolute and blocks self-directed callbacks. |
| `CALLBACK_ALLOWED_HOSTS` | unset | Comma-separated hosts `callback_url` may target. Unset = any host that passes the address policy. Leading dot matches subdomains. |
| `MAX_QUEUE_MB`      | `2048`    | Total queued audio on disk before `POST /jobs` returns 429. |
| `MODEL_DEFAULT`     | `small`   | Default whisper model when none supplied.                    |
| `MODEL_ALLOWLIST`   | see below | Comma-separated allowed model names.                         |
| `MAX_UPLOAD_MB`     | `25`      | Upload size cap.                                             |
| `MAX_QUEUE_DEPTH`   | `1000`    | Queue depth at which `POST /jobs` returns 429.               |
| `RETENTION_MINUTES` | `60`      | How long a completed job row is kept before purge.           |
| `WEBHOOK_ATTEMPTS`  | `5`       | Max webhook delivery attempts.                               |
| `WEBHOOK_BACKOFF`   | `2`       | Base backoff seconds (doubles each attempt).                 |
| `API_TOKEN`         | generated | Override to supply your own bearer token.                    |
| `WEBHOOK_SECRET`    | generated | Override to supply your own HMAC secret.                     |
| `ROTATE_SECRETS`    | `false`   | Force secret regeneration on boot.                            |
| `DATA_DIR`          | `/data`   | Volume mount for `jobs.db`, secrets, and temp audio.         |
| `LOG_LEVEL`         | `info`    | Logging verbosity.                                           |
| `VOCABULARY_HINT`   | unset     | Service-wide default hint, used when a job sends none.       |

| `MAX_VOCABULARY_HINT_CHARS` | `1000` | Cap on `vocabulary_hint`; longer values are rejected.  |

Copy [`.env.example`](.env.example) to `.env` for a documented starting point —
`.env` is gitignored, the template is not, so real secrets never get committed.

Default `MODEL_ALLOWLIST`: `tiny`, `base`, `small`, `medium` and their `.en`
variants. `MODEL_DEFAULT` must be a member of the allow-list.

Engine wiring, rarely changed: `MODEL_DIR` (`/models`), `WHISPER_SERVER_BIN`,
`WHISPER_CLI_BIN`, `WHISPER_HOST`, `WHISPER_PORT` (`8910`), `WHISPER_THREADS`
(`0` = let whisper.cpp decide), `WHISPER_STARTUP_TIMEOUT`,
`WHISPER_REQUEST_TIMEOUT`, `FFMPEG_BIN`, `WEBHOOK_TIMEOUT`,
`SWEEP_INTERVAL_SECONDS`.

TLS is deliberately **not** handled in-container: front the service with a reverse
proxy (Pangolin, Caddy, nginx) that terminates HTTPS and point it at
`http://<host>:8080`.

## Models in the image

Weights are downloaded at build time and baked in, so runtime is fully offline.
The published image carries `small` only. To bake more:

```bash
docker build --build-arg MODELS="small medium" -t echoback:with-medium .
```

Requesting an allow-listed model that is not baked in fails that job with
`MODEL_UNAVAILABLE`. The default model stays loaded in a resident
`whisper-server` child process; a non-default `model` override is transcribed with
a one-shot `whisper-cli` run, which is slower but keeps memory to one resident
model.

## Retention & privacy

- Uploaded audio is deleted from disk immediately after transcription, success or
  failure.
- The job row (including the transcript) is purged `RETENTION_MINUTES` after
  completion. That window exists only so the status endpoint and webhook retries
  work.
- Logs carry `job_id`, status, timings and error codes — never audio or transcript
  text.

## Operating notes

- **Concurrency is 1** by design. Bursts queue up in FIFO order; `MAX_QUEUE_DEPTH`
  is the backstop that keeps a runaway PBX from filling the disk.
- **Crash recovery:** a job left `processing` by a container that died is reset to
  `queued` on the next start and retried.
- **Secrets** live in `DATA_DIR/secrets.json` (mode 600), never in the image. Env
  `API_TOKEN` / `WEBHOOK_SECRET` take precedence and are not written to disk. Set
  `ROTATE_SECRETS=true` for one boot to regenerate and reprint.
- The container runs as an unprivileged user (uid 10001) with
  `CAP_NET_BIND_SERVICE` so the default `PORT=80` still binds. If your runtime
  drops capabilities, set `PORT` above 1024.

## Development

```bash
uv venv && uv pip install -e ".[dev]"   # or: python -m venv .venv && pip install -e ".[dev]"
.venv/bin/pytest -q                     # unit tests, no ffmpeg or models needed
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

The test suite stubs the ffmpeg and whisper.cpp binaries, so it runs anywhere. The
`docker-build` CI job exercises the real engine end to end: it builds the image with
the `tiny` model, submits an 8 kHz telephony-style WAV, and verifies the signed
callback.

### Checking transcript quality on real recordings

Synthetic audio proves the pipeline works; it says nothing about how well the model
handles your actual callers. Drop real voicemails into `tests/samples/` — the
directory is gitignored, so recordings never get committed — and run them through
the real engine:

```bash
./scripts/transcribe-samples.sh                 # uses the built image; no local deps
MODEL=medium ./scripts/transcribe-samples.sh    # needs that model baked into the image
pytest tests/test_samples.py -s                 # uses local ffmpeg + whisper.cpp instead
```

Each prints the file, audio duration, transcription time and transcript, so you can
compare models on the same recordings. The pytest route skips itself when the
binaries or weights are not on the machine. The same CLI works on any file:

```bash
python -m echoback.transcribe --model small --json voicemail.mp3
```

Run the service locally against real binaries with:

```bash
DATA_DIR=./data MODEL_DIR=./models PORT=8080 .venv/bin/python -m echoback.main
```

## License

MIT — see [LICENSE](LICENSE) and [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
ffmpeg is invoked as a CLI subprocess only, never linked, so its LGPL/GPL terms do
not extend to this code.
