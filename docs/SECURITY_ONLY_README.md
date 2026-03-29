# SECURITY ONLY README

Ce dépôt ne contient aucune logique relationnelle ni mémoire affective.
Il est limité à la surveillance résidentielle:
- détection d'événements vision
- gradation du danger avec hystérésis
- base locale de visages connus
- notifications Telegram optionnelles

## Flux local
1. `vision_agent` émet un `VisionEvent`
2. `alarm_grader_agent` reçoit `/ingest` (et `/sensor/camera` pour les captions RTSP)
3. `alarm_grader_agent` applique les seuils et l'hystérésis
4. `safetyagent` reste disponible pour `/evaluate` et notifications Telegram optionnelles

## Données
- `data/face_db.sqlite` contient les visages autorisés
- `models/yolov8n.onnx` doit être remplacé par un vrai modèle
- Les profils de config sont dans `config/security.local.toml`, `config/security.docker.toml`, `config/security.customer.toml`

## Validation source (b450)
- Le host source de vérité est b450.
- Validation complète: `scripts/validate_b450_source.sh`
- Smoke ciblé topology: `scripts/smoke_topology.sh`
