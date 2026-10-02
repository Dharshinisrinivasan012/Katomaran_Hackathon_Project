# Project Status

**Phase:** Phase 2 foundation (config + database)  
**Last updated:** 2026-10-02

Legend: `[x]` done · `[ ]` pending · `[-]` not applicable yet

---

## Planning & documentation

| Item | Planned | Implemented | Tested | Notes |
|------|---------|-------------|--------|-------|
| README.md | [x] | [x] | [-] | Aligned with official PDF |
| AI_PLANNING.md | [x] | [x] | [-] | SQLite-only confirmed |
| AI_PROMPTS.md | [x] | [x] | [-] | Prompts #1–#2 |
| ARCHITECTURE.md | [x] | [x] | [-] | Mermaid diagrams |
| config.json | [x] | [x] | [-] | Includes `detection_skip_frames` |
| requirements.txt | [x] | [x] | [-] | pytest listed |
| .gitignore | [x] | [x] | [-] | |
| Official problem statement PDF | [x] | [x] | [-] | `docs/katomaran-problem-statement.pdf` |
| Developer review of AI planning | [x] | [x] | [-] | **Completed** (Phase 2 kickoff) |
| Phase 1 planning | [x] | [x] | [-] | **Completed** |
| Phase 2 foundation (config + SQLite) | [x] | [x] | [x] | **Completed** — 14 pytest tests |
| Phase 3 (CV pipeline modules) | [x] | [ ] | [ ] | **Ready** — detection/tracking/recognition next |

---

## Core modules (implementation)

| Module | Planned | Implemented | Tested | Notes |
|------|---------|-------------|--------|-------|
| `src/config/` — load config.json | [x] | [x] | [x] | `CONFIG_PATH` env optional |
| `src/database/` — SQLite | [x] | [x] | [x] | Schema + repositories |
| `src/input/` — video & RTSP | [x] | [ ] | [ ] | Config-driven |
| `src/detection/` — YOLO faces | [x] | [ ] | [ ] | Skip-frame support |
| `src/tracking/` — ByteTrack | [x] | [ ] | [ ] | |
| `src/recognition/` — InsightFace | [x] | [ ] | [ ] | No face_recognition lib |
| `src/registration/` — auto-register | [x] | [ ] | [ ] | |
| `src/logging/` — events.log | [x] | [ ] | [ ] | |
| `src/pipeline/` — orchestration | [x] | [ ] | [ ] | |

---

## Features

| Feature | Planned | Implemented | Tested | Notes |
|---------|---------|-------------|--------|-------|
| Face detection (YOLO) | [x] | [ ] | [ ] | |
| Multi-object tracking (ByteTrack) | [x] | [ ] | [ ] | |
| Face embeddings (InsightFace) | [x] | [ ] | [ ] | |
| Auto-registration of unknown faces | [x] | [ ] | [ ] | |
| Re-identification (gallery match) | [x] | [ ] | [ ] | Threshold in config |
| Unique visitor count | [x] | [ ] | [ ] | DB visit helpers only |
| Entry / exit events (1:1 per visit) | [x] | [ ] | [ ] | Line crossing |
| Sample video input | [x] | [ ] | [ ] | |
| RTSP stream input | [x] | [ ] | [ ] | Reconnect policy in config |
| Crop storage on filesystem | [x] | [ ] | [ ] | |
| SQLite persistence | [x] | [x] | [x] | |

---

## Quality & ops

| Item | Planned | Implemented | Tested | Notes |
|------|---------|-------------|--------|-------|
| Unit tests (`tests/`) | [x] | [x] | [x] | config + database |
| Integration test on sample video | [x] | [ ] | [ ] | |
| CPU-only deployment path | [x] | [ ] | [ ] | Documented |
| GPU / ONNX CUDA path | [x] | [ ] | [ ] | Optional providers |

---

## Blockers / decisions open

- [x] Official problem statement PDF in repo (`docs/katomaran-problem-statement.pdf`)
- [ ] Final YOLO face weights filename and download source
- [ ] Entry/exit geometry calibration for deployment camera
- [ ] Similarity threshold tuning on target hardware
