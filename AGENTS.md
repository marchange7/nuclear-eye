# Nuclear Eye — Agent Rules

## Build Execution Policy (B450 Source of Truth)

- Source host for compile/test artifacts: `crew@192.168.2.23` (B450).
- On local M2 (`Darwin/arm64`), do not run heavy compile/test loops by default.
- Local override is explicit only: `FORCE_LOCAL=true`.
- Default path is remote sync/compile/sync-back.
- If work reaches `nuclear-consul`, stop and discuss before changes.

## Operating Notes

- Keep customer deployments private-first and in-house by default.
- Prefer reproducible scripts over ad-hoc one-off commands.
- Preserve logs and health endpoints when changing topology.
