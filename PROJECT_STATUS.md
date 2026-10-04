# Project Status

**Phase:** Phase 6 — Complete end-to-end pipeline  
**Last updated:** 2026-10-04

Legend: `[x]` done · `[ ]` pending · `[-]` not applicable yet

---

## Planning & documentation

| Item | Status | Notes |
|------|--------|-------|
| README.md | [x] | Complete — setup, demo command, architecture, sample output |
| AI_PLANNING.md | [x] | All phases documented |
| AI_PROMPTS.md | [x] | Prompts #1–#8 recorded |
| ARCHITECTURE.md | [x] | Full pipeline diagram, SQLite schema |
| config.json | [x] | All keys calibrated; version 1.0.0 |
| requirements.txt | [x] | Cleaned — ultralytics + insightface + onnxruntime + pytest |
| .gitignore | [x] | Models, videos, DB, logs, __pycache__, tmp/ excluded |
| Official problem statement PDF | [x] | `docs/katomaran-problem-statement.pdf` |
| Sample output | [x] | `sample_output/` — events.log, DB, entry/exit JPEGs, annotated frame |
| Phase 1–6 planning | [x] | **All phases completed** |

---

## Core modules

| Module | Done | Tested | Notes |
|--------|------|--------|-------|
| `src/config/` | [x] | [x] | Typed loader + validation; `CONFIG_PATH` env |
| `src/database/` | [x] | [x] | SQLite schema + full CRUD repositories |
| `src/input/` | [x] | [x] | File + RTSP via OpenCV; `create_frame_source` |
| `src/detection/` | [x] | [x] | `YoloFaceDetector` (yolov8n-face.pt); `DetectionSkipPolicy` |
| `src/tracking/` | [x] | [x] | `ByteTrackFaceTracker`; Kalman predict on skip frames |
| `src/recognition/` | [x] | [x] | `InsightFaceEngine` buffalo_l; 512-dim ArcFace; cosine matcher |
| `src/registration/` | [x] | [x] | `FaceRegistrar`; UUID face_id; gallery search; crop save |
| `src/logging/` | [x] | [x] | `setup_logging`; JSON-lines events.log; RotatingFileHandler |
| `src/pipeline/entry_exit.py` | [x] | [x] | Line-crossing FSM; implicit entry; track-lost flush |
| `src/pipeline/visit_manager.py` | [x] | [x] | ENTRY/EXIT → DB visits; snapshot save |
| `src/pipeline/main.py` | [x] | [x] | Full end-to-end CLI; `--video`, `--db-path`, `--max-frames`, `--save-dir` |

---

## Features

| Feature | Done | Tested |
|---------|------|--------|
| YOLO face detection | [x] | [x] |
| ByteTrack multi-face tracking | [x] | [x] |
| InsightFace 512-dim ArcFace embeddings | [x] | [x] |
| Auto-registration of unknown faces (UUID face_id) | [x] | [x] |
| Cosine gallery re-identification | [x] | [x] |
| Entry/exit FSM (line-crossing + implicit entry) | [x] | [x] |
| Exactly one ENTRY + EXIT per visit | [x] | [x] |
| Unique visitor count (persons table) | [x] | [x] |
| SQLite persistence (persons, embeddings, events, visits) | [x] | [x] |
| events.log JSON-lines audit trail | [x] | [x] |
| ENTRY/EXIT face crop snapshots | [x] | [x] |
| File video input (MP4) | [x] | [x] |
| RTSP stream input (config-driven) | [x] | [x] |
| Annotated frame output (`--save-dir`) | [x] | [x] |
| Isolated demo DB (`--db-path :memory:`) | [x] | [x] |

---

## Tests

| Suite | Count | Result |
|-------|-------|--------|
| test_config.py | 8 | ✅ |
| test_database.py | 8 | ✅ |
| test_detection.py | 9 | ✅ |
| test_input.py | 9 | ✅ |
| test_pipeline.py | 22 | ✅ |
| test_preview.py | 1 | ✅ |
| test_recognition.py | 29 | ✅ |
| test_tracking.py | 8 | ✅ |
| **Total** | **94** | **✅ 0 failures** |

---

## Verified real pipeline result (30 frames, fresh DB)

```
Frames processed      : 30
Detection frames      : 10
Total YOLO detections : 98
Embeddings extracted  : 70
New registrations     : 10
Re-identifications    : 60
Total entries (ENTRY) : 8
Total exits   (EXIT)  : 8
Unique visitors (DB)  : 10
Total visits (visits) : 8
Open visits remaining : 0
```

---

## Remaining (manual)

- [ ] Record demo video (Loom/YouTube) and update README link before submission
- [ ] Submit GitHub repo via hackathon Google Form before **12:00 PM, October 4, 2026**
