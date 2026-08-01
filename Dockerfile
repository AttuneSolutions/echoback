# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — build whisper.cpp (MIT) and bake the model weights in.
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS engine

ARG WHISPER_REF=v1.7.4
# Space-separated list of ggml models to bake into the image. Anything not baked
# is still accepted by MODEL_ALLOWLIST but fails with MODEL_UNAVAILABLE at runtime.
ARG MODELS="small"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${WHISPER_REF}" https://github.com/ggerganov/whisper.cpp.git .

# Static libs so the runtime stage needs only libgomp + libstdc++.
# GGML_NATIVE=OFF is deliberate. It probes the *build* CPU with -mcpu=native /
# -march=native, which (a) bakes the builder's instruction set into a published
# image and (b) breaks outright on the QEMU-emulated arm64 leg, where the probe
# cannot run and ggml emits an invalid -mcpu=native+nodotprod+noi8mm+nosve.
# With it off we get the portable baseline: armv8-a on arm64, AVX2/FMA/F16C on
# x86-64.
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DWHISPER_BUILD_TESTS=OFF \
    && cmake --build build --config Release -j "$(nproc)" \
    && cmake --install build --prefix /engine \
    && install -Dm755 build/bin/whisper-cli /engine/bin/whisper-cli \
    && install -Dm755 build/bin/whisper-server /engine/bin/whisper-server

RUN mkdir -p /models \
    && for model in ${MODELS}; do \
         bash ./models/download-ggml-model.sh "${model}" \
         && mv "models/ggml-${model}.bin" /models/; \
       done \
    && ls -lh /models

# Licences we must ship alongside the binaries and weights (§11).
RUN mkdir -p /engine/licenses && cp LICENSE /engine/licenses/whisper.cpp.LICENSE

# ---------------------------------------------------------------------------
# Stage 2 — runtime.
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="Echoback" \
      org.opencontainers.image.description="Offline voicemail transcription with a signed webhook callback" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/AttuneSolutions/echoback"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgomp1 libcap2-bin \
    && rm -rf /var/lib/apt/lists/*

COPY --from=engine /engine/bin/whisper-cli /usr/local/bin/whisper-cli
COPY --from=engine /engine/bin/whisper-server /usr/local/bin/whisper-server
COPY --from=engine /models /models
COPY --from=engine /engine/licenses /usr/share/licenses/echoback

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && rm -rf /root/.cache

COPY LICENSE THIRD_PARTY_LICENSES /usr/share/licenses/echoback/

# Run unprivileged. CAP_NET_BIND_SERVICE lets the default PORT=80 still bind;
# if your runtime drops capabilities, set PORT to something above 1024.
# /models stays root-owned and world-readable — chown -R'ing it would duplicate
# every model file into this layer, doubling the image size for no benefit.
RUN useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin echoback \
    && setcap 'cap_net_bind_service=+ep' "$(readlink -f "$(command -v python3)")" \
    && mkdir -p /data && chown echoback:echoback /data
USER echoback

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data \
    MODEL_DIR=/models \
    PORT=80

VOLUME ["/data"]
EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','80')+'/health', timeout=4)"

ENTRYPOINT ["python", "-m", "echoback.main"]
