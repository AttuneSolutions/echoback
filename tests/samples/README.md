# Sample recordings

Drop real voicemail files here (WAV PCM/ADPCM, MP3 — anything ffmpeg decodes) to
validate transcript quality against actual recordings.

**Everything in this directory except this README is gitignored.** Real voicemails
are personal data; they must never be committed.

Run them through the real engine:

```bash
./scripts/transcribe-samples.sh          # via the built image, no local deps
pytest tests/test_samples.py -s          # via local ffmpeg + whisper.cpp binaries
```

Both print the file, audio duration, transcription time and transcript. The pytest
route skips itself when the binaries or model weights are not on this machine.
