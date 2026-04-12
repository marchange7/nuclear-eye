# nuclear-eye — Operations Handbook

Practical ops guide for b450 and Jetson Orin deployments.

---

## Build on b450

nuclear-eye must be compiled on b450 (Pop!_OS 24.04, aarch64/x86-64). Do not compile on the M4 Mac for production binaries.

### From the Mac — full deploy

```bash
# From ~/git/nuclear-eye on the Mac:
bash scripts/deploy_b450.sh          # rsync + build only
bash scripts/deploy_b450.sh --start  # rsync + build + start all services
```

The script:
1. rsyncs source (excluding `target/`, `.git/`, `data/`, logs) to `b450:/data/git/nuclear-eye`
2. SSHs to b450 and runs `sudo bash deploy/install.sh` (cargo build --release + systemd unit install)
3. Optionally starts all services

### On b450 directly

```bash
ssh b450
cd /data/git/nuclear-eye

# Build
cargo build --release

# Install systemd units (first time or after adding new services)
sudo bash deploy/install.sh
```

After the first install, subsequent deploys only need `cargo build --release` — the units are already installed and enabled.

### OpenCV feature (RTSP direct capture)

RTSP capture via `CAMERA_URL` requires the `opencv` feature:

```bash
# On b450 — install libopencv-dev first
sudo apt install libopencv-dev
cargo build --release --features opencv
```

Without the feature, vision_agent falls back to HTTP snapshot or synthetic events.

---

## Python sidecars

The ArcFace and perceive sidecars are Python processes managed separately from the Rust binaries.

### ArcFace sidecar (face_embedding_service.py)

```bash
# On b450
cd /data/git/nuclear-eye
pip install fastapi uvicorn insightface onnxruntime numpy pillow requests

# Start (downloads buffalo_l model on first run, ~330MB)
python face_embedding_service.py
# Runs at :5555
```

With GPU acceleration (RTX 2060 on b450):

```bash
pip install onnxruntime-gpu
ONNX_PROVIDERS=CUDAExecutionProvider python face_embedding_service.py
```

Or let it auto-detect: if onnxruntime-gpu is installed and CUDA is available, it picks CUDAExecutionProvider automatically.

The ArcFace sidecar is managed as `nuclear-eye-face-embedding.service` on b450 (already running).

### Perceive service (perceive_service.py)

```bash
pip install fastapi uvicorn aiohttp httpx pydantic
python perceive_service.py
# Runs at :8091, proxies to FER (:5555), SER (:8105), vision (:8090)
```

---

## systemd service management

### Start / stop / restart

```bash
# Core pipeline (start in this order)
sudo systemctl start nuclear-eye-face-db
sudo systemctl start nuclear-eye-alarm-grader

# Remaining agents
sudo systemctl start nuclear-eye-iphone-sensor \
                     nuclear-eye-safety-aurelie \
                     nuclear-eye-safetyagent \
                     nuclear-eye-decision-agent \
                     nuclear-eye-actuator-agent \
                     nuclear-eye-vision-agent

# Stop all
sudo systemctl stop nuclear-eye-vision-agent \
                    nuclear-eye-alarm-grader \
                    nuclear-eye-face-db

# Restart a single service after a binary update
sudo systemctl restart nuclear-eye-alarm-grader
```

### After a binary update

The systemd units point to `target/release/<binary>` directly. After `cargo build --release`, restart the affected service:

```bash
sudo systemctl restart nuclear-eye-alarm-grader
# or for all:
sudo systemctl restart nuclear-eye-{face-db,alarm-grader,vision-agent,iphone-sensor,safety-aurelie,safetyagent,decision-agent,actuator-agent}
```

### View logs

```bash
# Follow alarm grader (most important)
journalctl -u nuclear-eye-alarm-grader -f

# Last 100 lines for face_db
journalctl -u nuclear-eye-face-db -n 100

# All nuclear-eye services together
journalctl -u 'nuclear-eye-*' -f

# Increase log verbosity
sudo systemctl edit nuclear-eye-alarm-grader
# Add: [Service]
#      Environment=RUST_LOG=debug
sudo systemctl restart nuclear-eye-alarm-grader
```

### Check service status

```bash
sudo systemctl status nuclear-eye-alarm-grader --no-pager -l

# Quick health check for all services
bash scripts/health_check_all.sh
```

---

## Environment file

All services load `/home/crew/.config/nuclear-eye/env` via `EnvironmentFile` in their unit files.

Edit it directly on b450:

```bash
ssh b450
sudo -u crew nano /home/crew/.config/nuclear-eye/env
```

Key variables to set on first deploy:

```bash
DATABASE_URL=postgres://crew:CHANGEME@127.0.0.1/nuclear_eye
HOUSE_JWT_TOKEN=CHANGEME
FORTRESS_URL=http://127.0.0.1:7700
FORTRESS_API_TOKEN=<token from /etc/nuclear/secrets.toml>
SIGNAL_TOKEN=                           # optional: Signal notification token
ALERT_LANG=fr                           # "fr" or "en"

# Real camera mode (RTSP):
# CAMERA_URL=rtsp://admin:pass@192.168.1.100:554/stream1
# CAMERA_FPS=5

# HTTP snapshot mode:
# CAMERA_SNAPSHOT_URL=http://m4:8085/snapshot

# FastVLM caption service:
FASTVLM_URL=http://127.0.0.1:8091

# ArcFace sidecar:
ARCFACE_URL=http://127.0.0.1:5555

# SMS/Signal High-alarm notifications (optional):
# COMMS_URL=http://127.0.0.1:9100
# COMMS_API_TOKEN=<chain-comms token>
# COMMS_ALERT_RECIPIENT=+15141234567

SITE_ID=b450
```

After editing, restart affected services to pick up the new values.

---

## Camera configuration

### cameras.toml

Used by the Python camera server (`scripts/camera_server.py`). Each entry:

```toml
[settings]
grab_interval_secs = 2
fastvlm_url = "http://127.0.0.1:8091"
fortress_url = "http://127.0.0.1:7700"
house_security_url = "http://127.0.0.1:8780"

[[cameras]]
id = "front-door"
name = "Front Door"
url = "rtsp://admin:password@192.168.2.100:554/stream1"
enabled = true

[[cameras]]
id = "backyard"
name = "Backyard"
url = "rtsp://admin:password@192.168.2.101:554/stream1"
enabled = false

# Local device (webcam or Continuity Camera)
[[cameras]]
id = "m4-webcam"
name = "M4 Built-in Camera"
url = "device://0"
enabled = true
```

URL schemes:
- `rtsp://user:pass@host:port/path` — RTSP stream (requires opencv feature in Rust, or Python opencv)
- `http://host/snapshot.jpg` — HTTP JPEG snapshot (polled at `grab_interval_secs`)
- `device://N` — local V4L2/AVFoundation device index

To add a camera: append a `[[cameras]]` block and set `enabled = true`. Restart the camera service or reload the Python server.

### Rust vision_agent camera mode

The Rust `vision_agent` uses environment variables, not `cameras.toml`:

```bash
# RTSP direct (requires opencv feature):
CAMERA_URL=rtsp://admin:pass@192.168.2.100:554/stream1
CAMERA_FPS=5

# HTTP snapshot:
SNAPSHOT_URL=http://m4:8085/snapshot
# or:
CAMERA_SNAPSHOT_URL=http://m4:8085/snapshot

# Dev mode (no real camera):
VISION_ALLOW_SYNTHETIC=true
```

Set these in `/home/crew/.config/nuclear-eye/env`.

---

## Alarm grading tuning

Thresholds and grader behavior are set in `config/security.toml`:

```toml
[alarm]
history_len = 20          # rolling event window for hysteresis
hysteresis_window = 5     # consecutive events required to change level
thresholds = [0.30, 0.50, 0.80]   # [Low, Medium, High] danger score thresholds
telegram_min_level = "medium"      # minimum level to send Telegram alert
```

**Raising the High threshold** (e.g. to `0.90`) reduces false High alarms at the cost of delayed escalation.

**Lowering the Low threshold** (e.g. to `0.20`) makes the system more sensitive to mild events.

**Increasing `hysteresis_window`** requires more consecutive events before a level change fires — reduces flapping on intermittent motion.

After editing `security.toml`, restart the alarm grader:

```bash
sudo systemctl restart nuclear-eye-alarm-grader
```

### Depth-based suppression (LiDAR)

Rules are compile-time constants in `src/bin/alarm_grader_agent.rs` → `depth_adjust_score()`. To tune:

| Rule | Parameter | Default |
|---|---|---|
| Pet/cat suppression | blob `height < 0.5m` | 0.5 m |
| Intimate zone amplify | `alert_zone = "intimate"`, multiplier | ×1.2 |
| Projected zone attenuate | `alert_zone = "projected"`, multiplier | ×0.85 |
| Fall → Critical | `fall_detected = true` | score = 1.0 |

To change these, edit the constants and rebuild.

### Consul deliberation for High alarms

High alarms trigger a Consul query with an 80 ms timeout (`CONSUL_TIMEOUT_MS`). If nuclear-consul is unreachable, the local grade stands. Consul URL:

```bash
CONSUL_URL=http://127.0.0.1:7710   # default
```

The Consul note is appended to `alarm.note` and forwarded to the WebSocket and La Rivière.

---

## Face database management

face_db runs at `:8087` and uses `data/face_db.sqlite` (path from `config/security.toml` → `app.face_db_path`).

### Add a person

```bash
# Step 1: Register the name in faces table
curl -X POST http://localhost:8087/faces \
  -H 'Content-Type: application/json' \
  -d '{"name": "andrzej", "embedding_hint": "owner", "authorized": true}'

# Step 2: Store their ArcFace embedding (base64 JPEG photo)
IMAGE_B64=$(base64 -w0 /path/to/photo.jpg)
curl -X POST http://localhost:8087/faces/embed \
  -H 'Content-Type: application/json' \
  -d "{\"image_b64\": \"$IMAGE_B64\", \"face_name\": \"andrzej\"}"
```

A face must exist in the `faces` table before its embedding can be stored.

### Search by image (biometric)

```bash
IMAGE_B64=$(base64 -w0 /path/to/query.jpg)
curl -X POST http://localhost:8087/faces/search-by-image \
  -H 'Content-Type: application/json' \
  -d "{\"image_b64\": \"$IMAGE_B64\", \"limit\": 5, \"min_similarity\": 0.20}"
```

Returns ranked matches with `similarity` (0–1) and `likely_match` (true when similarity ≥ 0.28, the ArcFace same-person threshold).

### List all faces

```bash
curl http://localhost:8087/faces
```

### GDPR export (metadata only, no embeddings)

```bash
curl http://localhost:8087/faces/gdpr-export
```

### Purge stale faces

Faces not matched within `FACE_RETENTION_DAYS` (default 30) are eligible for purge. Run manually or from a cron job:

```bash
curl -X POST http://localhost:8087/faces/purge
# Returns: {"deleted": N, "retention_days": 30}
```

Override retention period: set `FACE_RETENTION_DAYS=90` in the env file.

### Delete a specific face

```bash
# No DELETE endpoint yet — use sqlite3 directly
sqlite3 /data/git/nuclear-eye/data/face_db.sqlite \
  "DELETE FROM faces WHERE name = 'andrzej';"
# Embedding is cascade-deleted automatically.
```

---

## Connecting to Fortress

alarm_grader_agent connects to Fortress on two paths:

**1. La Rivière domain events** — POST to `FORTRESS_URL/v1/events` (500 ms timeout, fire-and-forget). No auth required by default; set `FORTRESS_API_TOKEN` if your Fortress instance requires a bearer token.

**2. Agent memory** — POST to `FORTRESS_URL/v1/agents/{emile,arianne}/memory` with bearer `FORTRESS_API_TOKEN`. Active alarm context is written on Medium/High, cleared on None/Low.

Verify Fortress connectivity:

```bash
curl http://localhost:7700/health
```

If Fortress is on a different host:

```bash
# In /home/crew/.config/nuclear-eye/env:
FORTRESS_URL=http://192.168.2.23:7700
FORTRESS_API_TOKEN=<your token>
```

---

## Operator feedback loop

Send feedback on an alarm decision to feed the Sentinelle learning pipeline:

```bash
curl -X POST http://localhost:8780/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "alarm_id": "<uuid from alarm event>",
    "camera_id": "front-door",
    "feedback": "false_alarm",
    "operator": "andrzej",
    "notes": "cat on porch"
  }'
```

`feedback` values: `false_alarm`, `confirmed`, `escalate`

This emits a `sentinelle.feedback` domain event to La Rivière for continuous learning.

---

## Audit log

Every graded verdict is appended to the audit log (JSON Lines):

```bash
tail -f /var/log/nuclear-eye/audit.jsonl | jq .
```

Override path: `AUDIT_LOG_PATH=/custom/path/audit.jsonl`

---

## Summary and WebSocket

```bash
# Current alarm summary (last N events)
curl http://localhost:8780/summary | jq .

# WebSocket stream (nuclear-watch connects here)
# ws://b450:8780/ws
# Events: alarm, decision, pedestrian, vision (JSON, tagged by "type" field)
```

---

## Smoke test / health check

```bash
# All services
bash scripts/health_check_all.sh

# Individual health endpoints
curl -s http://localhost:8780/health | jq .   # alarm_grader
curl -s http://localhost:8087/health | jq .   # face_db
curl -s http://localhost:8081/health | jq .   # safetyagent
curl -s http://localhost:8085/health | jq .   # decision_agent
curl -s http://localhost:8086/health | jq .   # safety_aurelie

# ArcFace sidecar
curl -s http://localhost:5555/health

# Send a synthetic alarm event
curl -X POST http://localhost:8780/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "event_id": "test-001",
    "timestamp_ms": 1700000000000,
    "camera_id": "front-door",
    "behavior": "person walking",
    "risk_score": 0.6,
    "stress_level": 0.4,
    "confidence": 0.85,
    "person_detected": true,
    "person_name": null,
    "hands_visible": 2,
    "object_held": null,
    "extra_tags": []
  }'
```

---

## Jetson Orin deployment

The Jetson deployment follows the same pattern as b450 with these differences:

- Build target: `aarch64-unknown-linux-gnu` (cross-compile from Mac or build natively on Jetson)
- ONNX providers: `CUDAExecutionProvider` is preferred (Jetson has integrated GPU)
- `cameras.toml` RTSP URLs point to Jetson-local cameras or VIGI NVR streams
- `SITE_ID=jetson-<id>` to distinguish events in La Rivière
- systemd units are identical; copy from `deploy/` and adjust `WorkingDirectory` and user as needed

For the VAR/Sentinelle appliance model, see `nuclear-eye/docs/` or the VAR Jetson strategy doc in the crew vault.

---

## Common issues

**vision_agent not sending events**
- Check `SNAPSHOT_URL` or `CAMERA_URL` is set and reachable
- Check `FASTVLM_URL` is reachable: `curl http://localhost:8091/health`
- Enable synthetic mode to test the pipeline without a camera: `VISION_ALLOW_SYNTHETIC=true`

**face_db returning empty search results**
- Confirm ArcFace sidecar is running: `curl http://localhost:5555/health`
- Confirm the face has an embedding stored (not just a `faces` row): `curl http://localhost:8087/faces/andrzej`; use `/faces/embed` to add the embedding
- Check `ARCFACE_URL` in the env file matches the sidecar port

**alarm_grader not emitting to La Rivière**
- `FORTRESS_URL` must point to a reachable Fortress instance
- Errors are logged at WARN, not ERROR — check `journalctl -u nuclear-eye-alarm-grader | grep -i rivière`
- La Rivière emission is non-blocking; alarm delivery continues regardless

**High alarm not triggering Consul deliberation**
- Check `CONSUL_URL` is reachable: `curl http://localhost:7710/health`
- Consul query has an 80 ms hard timeout; if consul is slow, the local grade is used
- Check consul logs: `journalctl -u nuclear-consul -n 50`

**Telegram alerts not sending**
- `telegram.enabled = true` must be set in `security.toml`
- `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` must be set in the env file
- `telegram_min_level` controls minimum alarm level that triggers a Telegram message
