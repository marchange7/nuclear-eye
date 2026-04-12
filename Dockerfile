# nuclear-eye/Dockerfile
# Multi-stage build for all nuclear-eye Rust service binaries.
#
# Build context: the git root (parent directory of nuclear-eye/).
# docker-compose must use:
#   build:
#     context: ..
#     dockerfile: nuclear-eye/Dockerfile
#
# The specific binary to run is chosen by the docker-compose `command:` field.
# All binaries are installed at /usr/local/bin/<name>.

# ── Stage 1: Builder ──────────────────────────────────────────────────────────

FROM rust:1.78-bullseye AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libssl-dev \
        libpq-dev \
        cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy local dependency crates first (separate layer for Docker cache).
# These change less often than nuclear-eye source.
COPY nuclear-sdk/               nuclear-sdk/
COPY nuclear-consul/            nuclear-consul/
COPY nuclear-wrapper/core/      nuclear-wrapper/core/
COPY nuclear-platform/fortress/crates/nuclear-voice-client/ \
     nuclear-platform/fortress/crates/nuclear-voice-client/

# Copy nuclear-eye source.
COPY nuclear-eye/               nuclear-eye/

# Build all release binaries.
WORKDIR /build/nuclear-eye
RUN cargo build --release --bins

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────

FROM debian:bullseye-slim

# Runtime library dependencies.
# - ca-certificates: for TLS (reqwest rustls uses system certs for chain validation)
# - libpq5: for nuclear-wrapper sqlx/postgres client
# - ffmpeg: for camera_server RTSP capture via subprocess
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libpq5 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy all service binaries.
COPY --from=builder /build/nuclear-eye/target/release/camera_server        /usr/local/bin/camera_server
COPY --from=builder /build/nuclear-eye/target/release/vision_agent         /usr/local/bin/vision_agent
COPY --from=builder /build/nuclear-eye/target/release/alarm_grader_agent   /usr/local/bin/alarm_grader_agent
COPY --from=builder /build/nuclear-eye/target/release/safetyagent          /usr/local/bin/safetyagent
COPY --from=builder /build/nuclear-eye/target/release/face_db              /usr/local/bin/face_db
COPY --from=builder /build/nuclear-eye/target/release/decision_agent       /usr/local/bin/decision_agent
COPY --from=builder /build/nuclear-eye/target/release/iphone_sensor_agent  /usr/local/bin/iphone_sensor_agent
COPY --from=builder /build/nuclear-eye/target/release/safety_aurelie_agent /usr/local/bin/safety_aurelie_agent
COPY --from=builder /build/nuclear-eye/target/release/actuator_agent       /usr/local/bin/actuator_agent

RUN mkdir -p /etc/nuclear /var/log/nuclear-eye

# No default ENTRYPOINT — docker-compose sets the binary via `command:`.
# e.g. command: camera_server  →  exec /usr/local/bin/camera_server
