# House Security AI — Architecture

## System Overview

```mermaid
graph TD
    CAM[🎥 Camera / RTSP] -->|frames| VA[VisionAgent]
    VA -->|VisionEvent JSON| AG[AlarmGraderAgent :8780]
    CAM -->|caption payload| AG
    AG -->|AlarmEvent| SA[SafetyAgent :8081]
    SA -->|if level ≥ min| TG[Telegram Bot]
    SA -->|AlarmEvent| SAA[SafetyAurélieAgent :8086]
    SAA -->|chat payload| AUR[Aurélie ChatAgent :8090]
    AUR -->|reply| SAA
    SAA -->|if telegram_on_alarm| TG
    VA -->|VisionEvent| DA[DecisionAgent :8085]
    DA -->|AffectTriad + DecisionAction| LOG[Structured Logs]

    FDB[FaceDB :8087] -.->|known/unknown| VA

    style VA fill:#4a9,stroke:#333,color:#fff
    style SA fill:#49a,stroke:#333,color:#fff
    style AG fill:#a94,stroke:#333,color:#fff
    style DA fill:#94a,stroke:#333,color:#fff
    style SAA fill:#a49,stroke:#333,color:#fff
```

## Agent Inventory

| Agent               | Binary                    | Default Port | Config Key / Env Override              |
|---------------------|---------------------------|-------------|----------------------------------------|
| VisionAgent         | `vision_agent`            | —           | `vision.target_url` (outbound)         |
| SafetyAgent         | `safetyagent`             | 8081        | `app.bind_safetyagent`                 |
| AlarmGraderAgent    | `alarm_grader_agent`      | 8780        | `app.bind_alarm_grader`                |
| FaceDB              | `face_db`                 | 8087        | `app.bind_face_db`                     |
| DecisionAgent       | `decision_agent`          | 8085        | `decision.bind` / `DECISION_AGENT_BIND`|
| SafetyAurélieAgent  | `safety_aurelie_agent`    | 8086        | `aurelie_bridge.bind` / `SAFETY_AURELIE_BIND` |

## Configuration

All agents read the config file from `HOUSE_SECURITY_CONFIG`.
Recommended profiles:
- Local: `config/security.local.toml`
- Docker: `config/security.docker.toml`
- Customer node: `config/security.customer.toml`

New config sections (backward-compatible, defaults apply if absent):

```toml
[decision]
bind = "0.0.0.0:8085"
safety_risk_threshold = 0.5     # VisionEvent.risk_score above this → safety-critical

[aurelie_bridge]
bind = "0.0.0.0:8086"
aurelie_chat_url = "http://127.0.0.1:8090/chat"
request_timeout_secs = 30
telegram_on_alarm = true        # Also push alarm+reply to Telegram
```

## AffectTriad Decision Flow

```mermaid
flowchart LR
    VE[VisionEvent] --> COMPUTE["AffectTriad::from_vision_event()
    J = 0.7×(1−stress) + 0.3×(1−risk)
    D = 0.6×(1−conf) + 0.4×risk
    Dt = 0.5×conf + 0.3×risk + 0.2×stress"]

    AE[AlarmEvent] --> COMPUTE2["AffectTriad::from_alarm_event()
    J = 0.6×(1−stress) + 0.4×(1−danger)
    D = 0.5×danger + 0.2×(1−severity) + 0.3×stress
    Dt = 0.5×severity + 0.3×danger + 0.2×stress"]

    COMPUTE --> CHECK{is_safety_critical?}
    COMPUTE2 --> CHECK

    CHECK -->|Yes| SAFETY["Dt > 0.7 ∧ D < 0.3 → Alarm
    Dt < 0.4 ∧ D > 0.6 → Pause
    else → None"]

    CHECK -->|No| RELATION["J > 0.7 ∧ D > 0.5 → Challenge
    J > 0.6 ∧ D < 0.4 → Support
    D > 0.7 ∧ J < 0.4 → Reassure
    Dt > 0.7 ∧ J > 0.5 → Support
    else → None"]
```

## Sequence: VisionEvent → AlarmEvent → Aurélie

```mermaid
sequenceDiagram
    participant C as Camera
    participant V as VisionAgent
    participant S as SafetyAgent
    participant G as AlarmGrader
    participant D as DecisionAgent
    participant A as SafetyAurélieAgent
    participant AU as Aurélie ChatAgent
    participant T as Telegram

    C->>V: video frame
    V->>G: POST /ingest (VisionEvent)
    V->>D: POST /decide (VisionEvent)
    D->>D: AffectTriad + decide()
    D-->>V: DecisionResponse

    G->>G: compute_danger_score + hysteresis
    G-->>S: AlarmEvent

    alt level ≥ telegram_min_level
        S->>T: sendMessage
    end

    S->>A: POST /alert (AlarmEvent)
    A->>A: AffectTriad::from_alarm_event + decide()
    A->>AU: POST /chat (AlarmEvent as MultiModalContext)
    AU-->>A: Aurélie reply

    alt telegram_on_alarm
        A->>T: sendMessage (alarm + reply + triad)
    end
    A-->>S: AlertResponse
```

## Error Handling

| Agent              | Failure Mode               | Behaviour                                |
|--------------------|---------------------------|------------------------------------------|
| DecisionAgent      | Bad JSON                  | 400 + structured error body              |
| SafetyAurélieAgent | Aurélie unreachable       | 1 retry after 500ms, then fallback text  |
| SafetyAurélieAgent | Telegram send fails       | Logged, non-blocking (fire-and-forget)   |
| All agents         | Primary config unreadable | Tries `.bak` backup before failing loud  |
| Both new agents    | SIGTERM / Ctrl+C          | Graceful shutdown, in-flight reqs finish |

## Integration with Aurélie

The `SafetyAurélieAgent` bridges security events into the relational AI:

1. Receives `AlarmEvent` from SafetyAgent
2. Derives `AffectTriad` from alarm stress/danger/severity
3. Calls `decide()` in safety-critical mode
4. Constructs an `AurelieMultiModalContext` and forwards to Aurélie's `/chat`
5. On success: returns Aurélie's empathetic text + sends Telegram
6. On failure (after retry): returns a localised fallback message

This enables the security system to produce **empathetic notifications** —
not just raw alarm data, but contextual, emotionally-aware messages.
