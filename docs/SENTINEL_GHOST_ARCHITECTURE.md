# Sentinel + Ghost Architecture

**Version:** 1.0  
**Date:** 2026-03-27  
**Status:** Canonical — decision flow locked

---

## Decision Pipeline

```
Sentinel caméra détecte visage
  ↓
face_db check  (routage — rapide, local)
  ↓ miss
ghost activé   (routage — envoie face)
  ↓
Ghost Council  (après routage — délibère)
  → reverse_img
  → osint_social
  → behavior
  → context
  → face_db
  ↓
verdict identité → retour Sentinel
  ↓
AffectTriad + décision action
```

---

## Ghost Council — Qui est cette personne ?

5 membres, vote pondéré. Ghost répond **"qui"**.

| # | Membre | Source | Question |
|---|--------|--------|----------|
| 1 | `face_db` | Mémoire locale | Déjà vu ? |
| 2 | `reverse_img` | Google Lens / Bing Visual | Visage connu sur le web ? |
| 3 | `osint_social` | LinkedIn, Instagram, Facebook | Profil public trouvé ? |
| 4 | `behavior` | Sentinel — analyse posture/mouvement | Comportement suspect ? |
| 5 | `context` | Heure, zone, récurrence | Contexte anormal ? |

**Verdict :**
- 3/5 confiants → identité probable → log silencieux
- 4/5 → notification Emile
- 5/5 → alerte + action directe

---

## AffectTriad — Quoi faire ?

Sentinel décide **"quoi faire"** avec le verdict d'identité.

| Jugement | Doute | Détermination | Action |
|----------|-------|---------------|--------|
| bas | élevé | bas | log + observe |
| moyen | moyen | moyen | alerte douce Emile |
| élevé | bas | élevé | alarme + notification |

---

## Mémoire des Visiteurs

- **Visite 1** — profil créé, doute élevé, décision prudente
- **Visite 2** — historique chargé, comportement comparé
- **Visite 3+** — pattern établi, détermination monte, action proactive

Inverse : visiteur fréquent + comportement calme → confiance croissante → moins d'alertes.

---

## Ghost — Règles d'activation

- **Périmètre seulement** — caméras intérieures n'activent pas Ghost
- **One-shot par inconnu** — une fois identifié, face_db prend le relais
- **Jamais en zone publique** — hors scope

```toml
# config/perimeter_zones.toml
[[camera]]
id = "front_door"
perimeter = true
ghost_enabled = true

[[camera]]
id = "living_room"
perimeter = false
ghost_enabled = false
```

---

## Cross-référence Mesh

Face connue dans **un** nœud → connue dans **tous** les nœuds via Fortress.

```
Sentinel (RPi)  ──┐
Eye     (iPhone) ─┼── Fortress ── face_db partagée
Emile   (crew)  ──┘
```

Crew member reconnu par Sentinel → aucune alerte, jamais.

---

## Repos

| Composant | Repo |
|-----------|------|
| Sentinel (caméras + AffectTriad) | `house-security-ai` |
| Ghost (OSINT + Council) | `nuclear-ghost` |
| Face DB partagée | via `nuclear-fortress` |
| Eye (outil portable) | `nuclear-eye` |
