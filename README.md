# Intelligent Face Tracker with Auto-Registration and Visitor Counting

Automated entrance monitoring: detect faces in video or RTSP streams, track individuals with ByteTrack, recognize or auto-register visitors via InsightFace embeddings, and count **unique visits** with paired entry and exit events stored in SQLite.

**Current phase:** Phase 2 foundation — config loader and SQLite layer implemented; CV pipeline not started. See [PROJECT_STATUS.md](PROJECT_STATUS.md).

---

## Problem statement

Official hackathon brief: **[docs/katomaran-problem-statement.pdf](docs/katomaran-problem-statement.pdf)**.

**Objective (from brief):** Build an AI-driven unique visitor counter on a video stream—detect, track, and recognize faces; auto-register new faces; log every entry and exit with a timestamped crop in the filesystem and metadata in the database; maintain an accurate **unique** visitor count.

**Sample video (development):** [Google Drive folder](https://drive.google.com/drive/folders/15YCN3CYb97GyIoNUV6NJxGNIf-rUBFUJ?usp=sharing) (provided in the problem statement). **Interview testing** uses a live **RTSP** stream instead of the file.

---

## Features (planned / in progress)

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for implementation status.

- Detect faces in real time (YOLO-based face detector).
- Maintain stable track IDs across frames (ByteTrack).
- Assign a persistent **Face ID** via embedding similarity (InsightFace); auto-register unknown faces after quality gates.
- Record **one entry and one exit per visit**; count unique visitors without incrementing on mere re-recognition.
- Persist persons, embeddings, events, and visits in **SQLite**; store face crops on the local filesystem.
- Support **sample video files** and **RTSP** cameras through `config.json`.

---

## Tech stack

| Layer | Choice |
|--------|--------|
| Language | Python 3.10+ |
| Detection | YOLO (Ultralytics), face-specific weights |
| Tracking | ByteTrack |
| Recognition | InsightFace → ONNX Runtime |
| Video I/O | OpenCV |
| Database | SQLite (stdlib `sqlite3`) |
| Configuration | JSON (`config.json`) |
| Logging | Python `logging` + `logs/events.log` |

**Recognition (per brief):** InsightFace (selected); ArcFace / Dlib-ArcFace are acceptable alternatives. **Do not use** the `face_recognition` Python library.

**Database:** **SQLite only** (chosen from the brief’s allowed options: SQLite, MongoDB, or PostgreSQL).

---

## Repository layout

```
D:\katomaran\
├── README.md
├── ARCHITECTURE.md
├── AI_PLANNING.md
├── AI_PROMPTS.md
├── PROJECT_STATUS.md
├── config.json
├── requirements.txt
├── .gitignore
├── src/
│   ├── input/          # Video file & RTSP frame sources
│   ├── detection/      # YOLO face detection (skip frames from config)
│   ├── tracking/       # ByteTrack association
│   ├── recognition/    # InsightFace embed & match
│   ├── registration/   # Auto-register unknown tracks
│   ├── database/       # SQLite access & migrations
│   ├── logging/        # App logging & event writers
│   ├── config/         # Load & validate config.json
│   └── pipeline/       # End-to-end orchestration
├── tests/
└── data/
    ├── samples/        # Sample videos for dev
    ├── crops/          # Ephemeral / debug crops
    ├── faces/          # Registered face images by Face ID
    ├── events/         # Optional entry/exit snapshots
    └── sqlite/         # face_tracker.db
├── logs/               # events.log (runtime)
└── models/             # Downloaded YOLO / ONNX weights (gitignored)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for diagrams and [AI_PLANNING.md](AI_PLANNING.md) for implementation notes.

---

## Configuration

All runtime tuning lives in **`config.json`** (not hardcoded). Important keys:

| Key | Purpose |
|-----|---------|
| `input.source_type` | `"video"` or `"rtsp"` |
| `input.video_path` / `input.rtsp_url` | Source paths |
| `detection.detection_skip_frames` | Run detector every N+1 frames; track on intermediate frames |
| `detection.confidence_threshold` | Min detection score |
| `recognition.similarity_threshold` | Cosine match for re-id |
| `registration.auto_register_unknown` | Enable auto-enrollment |
| `entry_exit.*` | Virtual lines and debounce for visit pairing |
| `database.sqlite_path` | DB file location |
| `logging.file_path` | Typically `logs/events.log` |

Copy and adjust paths for your machine before running (implementation phase).

---

## SQLite overview

- **`persons`** — one row per unique Face ID (visitor identity).
- **`face_embeddings`** — embedding vectors linked to `persons` (versioned, deduplicated).
- **`events`** — append-only log: REGISTER, RECOGNIZE, ENTRY, EXIT, etc.
- **`visits`** — one row per completed or in-progress visit; enforces one entry + one exit pairing.

Full DDL and rationale: [ARCHITECTURE.md](ARCHITECTURE.md#sqlite-schema) and [AI_PLANNING.md](AI_PLANNING.md).

---

## Entry / exit and visitor counting (summary)

1. **Tracking** assigns a `track_id` per face in the scene.
2. **Recognition** maps stable tracks to a **`face_id`** (existing or new registration).
3. **Line crossing** (configurable) emits at most one **ENTRY** per visit when the track crosses the entry line in the configured direction; **EXIT** when crossing the exit line.
4. **Unique visitors:** `persons.total_visits` (or equivalent) increments only when a **new visit** starts (ENTRY for a face not already in an open visit), not when the same person is recognized again on subsequent frames or days without a new visit policy trigger.
5. Re-identification links returning visitors to the same `face_id` without creating duplicate person rows.

Details: [AI_PLANNING.md](AI_PLANNING.md#entry-exit-and-visit-counting).

---

## Data flow (summary)

```
Input (video/RTSP) → frames → [skip] Detection → boxes
  → Tracking (ByteTrack) → track states
  → Recognition (crop → embed → match/register) → face_id
  → Entry/exit logic → events + visits
  → SQLite + filesystem crops + events.log
```

---

## Setup (after implementation phase)

1. Create a virtual environment and install dependencies:  
   `pip install -r requirements.txt`  
   (Planning phase does **not** run install automatically.)
2. Download YOLO face weights and InsightFace models into `models/` per deployment docs.
3. Edit `config.json` for input source and hardware (`detection.device`, `recognition.providers`).
4. Run the pipeline entrypoint (TBD in `src/pipeline/`).

Optional: set `CONFIG_PATH` to load a non-default `config.json`.

---

## Logging

- **Python logging:** module loggers, level from config.
- **`logs/events.log`:** structured business events (entry, exit, register, errors) for audit and debugging.

---

## CPU / GPU

- **CPU:** Default ONNX CPU provider; smaller YOLO model; increase `detection_skip_frames` to reduce load.
- **GPU:** CUDA-enabled PyTorch/ONNX where available; set `detection.device` and `recognition.providers` accordingly.

See [AI_PLANNING.md](AI_PLANNING.md#cpu-and-gpu-considerations).

---

## Edge cases (high level)

Occlusion, missed detections, track ID switches, duplicate registration, RTSP disconnects, multiple faces in one track, and rapid line bouncing — mitigations documented in [AI_PLANNING.md](AI_PLANNING.md#edge-cases).

---

## Sample output (submission)

Per the problem statement, the final submission must include sample output from the sample video: **`logs/events.log`**, entry/exit **cropped images**, and **SQLite** rows. Placeholders until the CV pipeline runs in a later phase.

---

## Assumptions

- **SQLite** is used as the sole database (explicit team choice under the brief).
- **ByteTrack** is used for tracking (brief allows OpenCV / DeepSort / ByteTrack).
- **Entry/exit** will be implemented so each face gets exactly one entry and one exit event per appearance; geometry is configurable (`entry_exit` in `config.json`)—including line-based zones or frame-boundary detection aligned with “entering/exiting the camera frame” in the brief.
- Entry/exit snapshot folders are configurable; the brief suggests a structure such as `logs/entries/YYYY-MM-DD/`—this project uses `paths.events_snapshot_dir` (see `config.json`) unless adjusted at demo time.
- A **demo video** (Loom or YouTube) explaining and demonstrating the solution is **required for submission review** (see below).

---

## Testing

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

Integration testing on the sample video and RTSP is planned for later phases.

---

## AI usage

AI-assisted development is encouraged by the brief. Prompts used for important steps are recorded in [AI_PROMPTS.md](AI_PROMPTS.md) for interview review.

---

## Demo video (submission)

<!-- Required by Katomaran problem statement — submission may not be reviewed without this -->
**Demo link (Loom or YouTube):** _TBD — required for submission review per the problem statement._

Submit the GitHub repository via the hackathon Google Form before **12:00 PM, Sunday, October 4, 2026** (per the problem statement).

---

## Documentation index

| Document | Contents |
|----------|----------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System context, Mermaid diagrams, schema |
| [AI_PLANNING.md](AI_PLANNING.md) | Technical plan, strategies, phases |
| [AI_PROMPTS.md](AI_PROMPTS.md) | AI prompt log |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Checklist |

---

## License / attribution

Hackathon submission for Katomaran. Add license text when team decides.

This project is a part of a hackathon run by https://katomaran.com
