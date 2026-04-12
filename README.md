# nuclear-eye

Camera/vision server for the Sentinelle security product. Runs on b450 (192.168.2.23) and Jetson Orin as a set of Rust binaries + Python sidecars under systemd.

Part of the **Sentinelle** product family: nuclear-eye (server) + nuclear-scout (iOS sensor) + nuclear-watch (iOS viewer).

---

## What it does

nuclear-eye ingests camera frames, runs AI scene captioning and face recognition, grades incoming events against configurable danger thresholds, and emits structured domain events to Fortress (La Rivière) in real time.

Key capabilities:

- **RTSP / HTTP / device capture** — direct RTSP stream grab (OpenCV feature), HTTP snapshot polling, or local device index (`device://0`)
- **FastVLM scene captioning** — LLM-grade scene descriptions from every frame, served by a local Python sidecar
- **Alarm grading** — continuous danger scoring with hysteresis, multi-tier thresholds, and Consul multi-agent deliberation for High alarms
- **LiDAR depth integration** — nuclear-scout sends `depth_summary` alongside vision events; `depth_adjust_score()` suppresses cat/pet false positives, forces Critical on falls, and amplifies intimate-zone proximity threats
- **ArcFace face recognition** — 512-dim biometric embeddings via `face_embedding_service.py` at `:5555`; cosine similarity search with a `0.28` same-person threshold
- **La Rivière emission** — every graded event fans out to Fortress `/v1/events` (domain schema) and the nuclear-sdk stream bridge
- **nuclear-watch WebSocket** — alarm, decision, pedestrian, and vision events broadcast live to the iOS viewer
- **Fortress agent memory** — Medium/High alarms write `active_alarm` context to Emile and Arianne agents at Fortress `:7700`
- **GDPR retention** — face records purged after `FACE_RETENTION_DAYS` (default 30) days of inactivity; `/faces/gdpr-export` returns metadata only (no embeddings)

---

## Architecture

```
nuclear-scout (iOS)
  └─ sensor + LiDAR depth_summary
       │
       ▼ POST /sensor/iphone
iphone_sensor_agent :8089
       │
       ▼ VisionEvent
       ┌───────────────────────────────────────────────────────┐
       │               alarm_grader_agent :8780                │
       │  grade_event() → depth_adjust_score() → AlarmLevel    │
       │  High alarm → Consul :7710 + penny-brain :8000        │
       │  Fan-out: La Rivière · WebSocket · Fortress memory    │
       └──────────────┬──────────────┬───────────────┬─────────┘
                      │              │               │
              vision.* events  ws://…/ws      active_alarm
                      │              │          (Emile/Arianne)
              Fortress :7700   nuclear-watch        │
              La Rivière                      Fortress :7700

vision_agent (Rust)
  ├─ RTSP frame grab (opencv feature) or HTTP snapshot
  ├─ FastVLM caption → POST /ingest to alarm_grader
  └─ Synthetic events (VISION_ALLOW_SYNTHETIC=true, dev mode)

face_embedding_service.py :5555   (ArcFace buffalo_l, 512-dim)
face_db :8087                     (SQLite, cosine similarity)

perceive_service.py :8091         (multimodal: FER + SER + pose)
safetyagent :8081                 (Telegram notifications)
safety_aurelie_agent :8086        (Aurelia /api/safety bridge)
decision_agent :8085              (Consul-backed deliberation)
actuator_agent                    (MQTT actuator bridge)
house_sentinel / house_runtime    (identity-gated action guard)
```

---

## Services and ports

| Service | Binary / Script | Port | Description |
|---|---|---|---|
| alarm_grader_agent | `alarm_grader_agent` | 8780 | Core grading pipeline, La Rivière, WebSocket |
| vision_agent | `vision_agent` | — | Frame capture → captioning → ingest |
| face_db | `face_db` | 8087 | ArcFace identity store (SQLite) |
| face_embedding_service | `face_embedding_service.py` | 5555 | ArcFace sidecar (insightface buffalo_l) |
| iphone_sensor_agent | `iphone_sensor_agent` | 8089 | nuclear-scout pedestrian/LiDAR receiver |
| safetyagent | `safetyagent` | 8081 | Telegram alert dispatcher |
| safety_aurelie_agent | `safety_aurelie_agent` | 8086 | Aurelia safety bridge |
| decision_agent | `decision_agent` | 8085 | Consul-backed deliberation |
| actuator_agent | `actuator_agent` | — | MQTT actuator commands |
| perceive_service | `perceive_service.py` | 8091 | Multimodal perception (FER + SER) |
| house_runtime | `house-runtime` | — | Identity-gated protected actions |
| house_sentinel | `house-sentinel` | — | House perimeter sentinel |

---

## Alarm grading

Danger scores map to levels using three configurable thresholds (default `[0.30, 0.50, 0.80]`):

| Score range | Level |
|---|---|
| < 0.30 | None |
| 0.30 – 0.50 | Low |
| 0.50 – 0.80 | Medium |
| ≥ 0.80 | High |

High alarms trigger parallel Consul deliberation (80 ms timeout) and penny-brain query (500 ms timeout). The local grade stands if neither responds in time.

### Depth-enhanced scoring (LiDAR, JJ6)

When nuclear-scout sends `depth_context` alongside a vision event, `depth_adjust_score()` applies these rules in priority order:

1. All blobs `height < 0.5m` → suppress alarm (cat/pet, score forced to 0.0)
2. `fall_detected = true` → Critical (score forced to 1.0)
3. `alert_zone = "intimate"` (< 0.45m) → score × 1.2 (capped at 1.0)
4. `alert_zone = "projected"` (> 3.6m) → score × 0.85
5. Single occupant in intimate zone → additional × 1.1

Depth adjustment is logged at DEBUG. Suppressions are logged at INFO with reason.

---

## La Rivière integration

All events are fire-and-forget (spawned in `tokio::spawn`). Errors logged at WARN, never propagated.

Domain event types emitted:

| Type | Trigger |
|---|---|
| `vision.person_detected` | `person_detected = true` on any event |
| `vision.behavior_alert` | Every graded event |
| `vision.scene_captured` | VLM caption present |
| `vision.face_identified` | ArcFace match above threshold |
| `sentinelle.alarm` | Every alarm (includes depth context for learning pipeline) |
| `sentinelle.feedback` | Operator feedback via `/feedback` |
| `sentinelle.face` | ArcFace identity match |

Fortress endpoint: `FORTRESS_URL/v1/events` (default `http://localhost:7700`). 500 ms timeout per POST.

---

## Fortress integration

- `FORTRESS_URL` — base URL for Fortress (default `http://localhost:7700`)
- Active alarm context written to `/v1/agents/emile/memory` and `/v1/agents/arianne/memory` as key `active_alarm` on Medium/High; cleared (set to null) on None/Low
- `FORTRESS_API_TOKEN` — bearer token for Fortress memory API

---

## Sentinelle product context

nuclear-eye is the back-end vision server for the **Sentinelle** home security product.

```
Sentinelle
├── nuclear-eye      — vision server (this repo), runs on b450 / Jetson Orin
├── nuclear-scout    — iOS 18 Swift 6 sensor app (LiDAR, pedestrian AR overlay)
└── nuclear-watch    — iOS SwiftUI monitoring viewer (cameras, timeline, alerts)
```

nuclear-scout sends pedestrian + LiDAR depth summaries to `iphone_sensor_agent :8089`. nuclear-watch connects to the alarm_grader WebSocket at `ws://<b450>:8780/ws` for live event streaming.

---

## Tech stack

- **Rust 2021** — Axum 0.7, Tokio 1.37, SQLite (rusqlite bundled), MQTT (rumqttc), rustls
- **Python 3.11** — FastAPI/uvicorn sidecars (ArcFace, perceive, FastVLM)
- **insightface** — ArcFace buffalo_l R100, 512-dim embeddings
- **nuclear-sdk** — Fortress + penny-brain + consul feature flags
- **nuclear-wrapper** — resilience sidecar embedded in every binary
- **nuclear-consul** — multi-agent Consul deliberation for High alarms

Dependencies: `nuclear-sdk`, `nuclear-consul`, `nuclear-wrapper-core`, `nuclear-voice-client`

---

## Configuration

Primary config: `config/security.toml` (or `HOUSE_SECURITY_CONFIG` env var)

Camera list: `cameras.toml` (used by the Python camera server; Rust agents use `CAMERA_URL` / `SNAPSHOT_URL` env vars)

Environment file on b450: `/home/crew/.config/nuclear-eye/env`

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `HOUSE_SECURITY_CONFIG` | `config/security.toml` | Main config path |
| `FORTRESS_URL` | `http://localhost:7700` | Fortress base URL |
| `FORTRESS_API_TOKEN` | — | Bearer token for Fortress memory API |
| `CAMERA_URL` | — | RTSP URL for direct capture (requires opencv feature) |
| `SNAPSHOT_URL` | — | HTTP JPEG snapshot URL |
| `FASTVLM_URL` | `http://127.0.0.1:8091` | FastVLM caption service |
| `VISION_ALLOW_SYNTHETIC` | `false` | Enable synthetic events (dev/test) |
| `CAMERA_FPS` | `5` | RTSP frame grab rate |
| `VISION_TICK_MS` | `2500` | VLM call interval |
| `CONSUL_URL` | `http://127.0.0.1:7710` | nuclear-consul URL |
| `ARCFACE_URL` | `http://localhost:5555` | ArcFace sidecar URL |
| `FACE_RETENTION_DAYS` | `30` | GDPR face retention period |
| `ALERT_LANG` | `fr` | Alert message language |
| `COMMS_URL` | — | chain-comms SMS/Signal URL |
| `COMMS_ALERT_RECIPIENT` | — | E.164 phone/Signal number for High alerts |
| `SITE_ID` | — | Site identifier tag on La Rivière events |
| `AUDIT_LOG_PATH` | `/var/log/nuclear-eye/audit.jsonl` | Verdict audit log |
| `BIND_HOST` | `127.0.0.1` | Bind host (use `0.0.0.0` in Docker) |

---

> See [HOWTO.md](HOWTO.md) for development setup guide.
