# Intelligent Face Tracker with Auto-Registration and Visitor Counting

Automated entrance monitoring: detect faces in video or RTSP streams, track individuals with ByteTrack, recognize or auto-register visitors via InsightFace embeddings, and count **unique visits** with paired entry and exit events stored in SQLite.

**Status:** Complete end-to-end pipeline implemented and verified. All 94 tests pass.

---

## Problem statement

Official hackathon brief: **[docs/katomaran-problem-statement.pdf](docs/katomaran-problem-statement.pdf)**

**Objective (from brief):** Build an AI-driven unique visitor counter on a video stream — detect, track, and recognize faces; auto-register new faces; log every entry and exit with a timestamped crop in the filesystem and metadata in the database; maintain an accurate **unique** visitor count.

### Official sample video source

Development sample videos: **https://drive.google.com/drive/folders/15YCN3CYb97GyIoNUV6NJxGNIf-rUBFUJ?usp=sharing**

Large media files are not committed to git. After downloading, set `input.video_path` in `config.json`.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place YOLO face weights at models/yolov8n-face.pt
#    Download: https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt
#    InsightFace buffalo_l (~280 MB) downloads automatically on first run.

# 3. Set your video path in config.json  OR  pass it on the CLI:
python -m src.pipeline.main --video data/videos/video_sample1.mp4 --no-display --max-frames 30

# 4. For a fully isolated fresh demo (no DB side-effects):
python -m src.pipeline.main --no-display --max-frames 30 --save-dir tmp/demo --db-path :memory:

# 5. Full video, save annotated frames:
python -m src.pipeline.main --no-display --save-dir tmp/run
```

---

## Features

| Feature | Status |
|---------|--------|
| YOLO face detection | ✅ `yolov8n-face.pt` via Ultralytics |
| ByteTrack multi-face tracking | ✅ Stable track IDs across skipped frames |
| InsightFace ArcFace embeddings | ✅ `buffalo_l`, 512-dim, L2-normalized |
| Auto-registration of new faces | ✅ UUID `face_id`, SQLite + crop saved |
| Re-identification (gallery match) | ✅ Cosine similarity, threshold 0.45 |
| Entry / exit detection | ✅ Line-crossing FSM, configurable geometry |
| Unique visitor counting | ✅ One row per person; increments only on new ENTRY |
| SQLite persistence | ✅ persons, embeddings, events, visits |
| Event audit log (`events.log`) | ✅ JSON-lines, rotating file |
| Entry/exit face crop snapshots | ✅ `data/events/YYYY-MM-DD/` |
| File video input | ✅ OpenCV, config-driven |
| RTSP stream input | ✅ Config-driven, reconnect backoff |
| Configurable everything | ✅ All thresholds/paths in `config.json` |

---

## Tech stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.10+ |
| Detection | YOLO (Ultralytics) — `yolov8n-face.pt` |
| Tracking | ByteTrack (via `ultralytics.trackers`) |
| Recognition | InsightFace `buffalo_l` → ArcFaceONNX |
| Embeddings | ONNX Runtime (CPU; CUDA optional) |
| Video I/O | OpenCV |
| Database | SQLite (`sqlite3` stdlib) |
| Configuration | JSON (`config.json`) |
| Logging | Python `logging` + `logs/events.log` |
| Tests | pytest (93 tests) |

---

## Repository layout

```
D:\katomaran\
├── README.md               # This file
├── ARCHITECTURE.md         # System diagrams and schema
├── AI_PLANNING.md          # Technical plan and phase notes
├── AI_PROMPTS.md           # AI prompt log (hackathon requirement)
├── PROJECT_STATUS.md       # Completion checklist
├── config.json             # All runtime configuration
├── requirements.txt
├── .gitignore
├── src/
│   ├── config/             # Load & validate config.json
│   ├── database/           # SQLite schema + repositories
│   ├── input/              # Video file & RTSP frame sources
│   ├── detection/          # YOLO face detection + skip-frame policy
│   ├── tracking/           # ByteTrack adapter
│   ├── recognition/        # InsightFace embed + cosine gallery match
│   ├── registration/       # Auto-register unknown faces
│   ├── logging/            # setup_logging() + events.log writer
│   ├── pipeline/
│   │   ├── entry_exit.py   # Line-crossing FSM (per face_id)
│   │   ├── visit_manager.py# Open/close visits, save snapshots
│   │   └── main.py         # Full end-to-end pipeline entry point
│   ├── preview_detections.py   # Dev: YOLO boxes only
│   ├── preview_tracking.py     # Dev: YOLO + ByteTrack IDs
│   └── preview_recognition.py  # Dev: YOLO + ByteTrack + InsightFace
├── tests/                  # 93 pytest tests
├── data/
│   ├── crops/              # Registration face crops
│   ├── events/             # Entry/exit snapshots (YYYY-MM-DD/)
│   ├── faces/              # Registered face images by face_id
│   ├── sqlite/             # face_tracker.db
│   └── videos/             # Local sample video (gitignored)
├── models/                 # YOLO .pt weights (gitignored)
├── logs/                   # events.log (runtime)
└── sample_output/          # Sample output for submission review
```

---

## Configuration (`config.json`)

All runtime values are configurable. Key settings:

| Key | Default | Purpose |
|-----|---------|---------|
| `input.source_type` | `"file"` | `"file"` or `"rtsp"` |
| `input.video_path` | `"data/videos/video_sample1.mp4"` | Local video path |
| `input.rtsp_url` | `""` | RTSP URL for live camera |
| `detection.model_path` | `"models/yolov8n-face.pt"` | YOLO weights (not in git) |
| `detection.confidence_threshold` | `0.5` | Min face detection score |
| `detection.detection_skip_frames` | `2` | Detect every N+1 frames |
| `recognition.similarity_threshold` | `0.45` | Cosine match for re-ID |
| `recognition.providers` | `["CPUExecutionProvider"]` | ONNX providers |
| `registration.auto_register_unknown` | `true` | Auto-enroll new faces |
| `entry_exit.entry_line` | `[[0.0,0.12],[1.0,0.12]]` | Normalized entry line (top of frame) |
| `entry_exit.exit_line` | `[[0.0,0.90],[1.0,0.90]]` | Normalized exit line (bottom of frame) |
| `entry_exit.direction_entry` | `"top_to_bottom"` | Crossing direction |
| `entry_exit.debounce_sec` | `1.0` | Min seconds between events |
| `entry_exit.min_track_age_frames` | `3` | Min confirmed frames before ENTRY |
| `database.sqlite_path` | `"data/sqlite/face_tracker.db"` | DB location |
| `logging.file_path` | `"logs/events.log"` | Audit log location |
| `paths.events_snapshot_dir` | `"data/events"` | Entry/exit crop directory |

### RTSP / live camera

```json
"input": {
  "source_type": "rtsp",
  "rtsp_url": "rtsp://user:pass@host:554/stream"
}
```

Then run `python -m src.pipeline.main --no-display`.

---

## Pipeline entry point

```
python -m src.pipeline.main [OPTIONS]

Options:
  --config PATH      Path to config.json (default: project root config.json)
  --video PATH       Override input video path (skips config input.video_path)
  --max-frames N     Stop after N frames (0 = full video)
  --save-dir DIR     Save annotated frames as JPEGs to DIR
  --no-display       Headless mode — no GUI window (use on servers / CI)
  --db-path PATH     Override SQLite path (use :memory: for clean isolated demo)
```

The pipeline runs:

```
Video/RTSP input
  → YOLO face detection (every detection_skip_frames+1 frames)
  → ByteTrack → temporary track_id per face
  → InsightFace buffalo_l → 512-dim embedding
  → Gallery search → existing face_id (re-ID) or new registration
  → EntryExitFSM → ENTRY / EXIT line-crossing events
  → VisitManager → open/close visits in SQLite, save face snapshots
  → events.log (JSON-lines per event)
  → Final summary printed to console
```

End-of-run summary (verified on `data/videos/video_sample1.mp4`, 30 frames):

```
============================================================
  INTELLIGENT FACE TRACKER — RUN SUMMARY
============================================================
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
============================================================
  Database  : :memory:  (use --db-path for persistent)
  Events log: logs/events.log
  Snapshots : data/events/YYYY-MM-DD/
============================================================
```

---

## Identity rules

| Identity | Source | Lifetime | Usage |
|----------|--------|----------|-------|
| `track_id` | ByteTrack | **Temporary** — resets when person re-enters frame | Motion association only |
| `face_id` | InsightFace gallery | **Persistent** — survives track resets; stored in SQLite | All identity decisions |

A new ByteTrack `track_id` never creates a new `face_id`. Recognition (cosine similarity against the embedding gallery) decides identity.

---

## Entry / exit and unique visitor counting

- **ENTRY** fires when a face's bbox bottom-centre crosses the configured entry line in the configured direction, after `min_track_age_frames` frames and `debounce_sec` cooldown.
- **EXIT** fires on exit line crossing. If a track is lost for more than `track_buffer` frames while the face is INSIDE, EXIT is auto-issued.
- Exactly **one ENTRY and one EXIT** per visit (enforced by `open_visit` / `close_visit` in the database layer).
- **`persons.total_visits`** increments only on a new ENTRY — never on re-recognition of the same face.
- **Unique visitors** = number of rows in `persons` table (one row per persistent `face_id`).

---

## SQLite schema

| Table | Purpose |
|-------|---------|
| `persons` | One row per unique visitor (`face_id`, `total_visits`, timestamps) |
| `face_embeddings` | 512-dim float32 blobs; SHA-256 deduplication |
| `events` | Append-only audit log (REGISTER, RECOGNIZE, ENTRY, EXIT, …) |
| `visits` | One row per visit; `entry_event_id` + `exit_event_id` pairing |
| `track_sessions` | Optional: track_id → face_id mapping for debugging |
| `daily_stats` | Optional: materialized daily counters |

Full DDL: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Logging

- **Console:** INFO level by default (config: `logging.level`).
- **`logs/events.log`:** JSON-lines per business event, rotating at 10 MB.

Sample log lines:
```json
{"event":"REGISTER","face_id":"FT-abc123...","track_id":1,"frame":0,"ts":"2026-10-04T04:20:00+00:00"}
{"event":"RECOGNIZE","face_id":"FT-abc123...","track_id":1,"sim":0.8954,"frame":3,"ts":"..."}
{"event":"ENTRY","face_id":"FT-abc123...","track_id":1,"visit_id":1,"frame":57,"snapshot":"data/events/2026-10-04/entry_abc123.jpg","ts":"..."}
{"event":"EXIT","face_id":"FT-abc123...","track_id":1,"visit_id":1,"frame":79,"snapshot":"data/events/2026-10-04/exit_abc123.jpg","ts":"..."}
{"event":"SYSTEM","message":"pipeline finished","frames":90,"unique_visitors":15,"entries":2,"exits":2,"ts":"..."}
```

---

## Development previews

```bash
# YOLO detection only
python -m src.preview_detections --no-display --max-frames 30 --save-dir tmp/preview

# ByteTrack tracking
python -m src.preview_tracking --no-display --max-frames 30 --save-dir tmp/preview_tracking

# Recognition + registration (ephemeral DB)
python -m src.preview_recognition --no-display --max-frames 30 \
    --save-dir tmp/preview_recognition --db-path :memory:
```

---

## Testing

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

**94 tests, 0 failures.** Coverage: config, database, input, detection, tracking, recognition, pipeline (entry/exit FSM + visit manager). Integration tests that require local YOLO weights or InsightFace buffalo_l are run only when those files exist.

---

## CPU / GPU

- **CPU (default):** `CPUExecutionProvider` for InsightFace; `detection.device: "auto"`. Increase `detection_skip_frames` (e.g. 4) to reduce load on slower hardware.
- **GPU:** Install `onnxruntime-gpu`. Set `detection.device: "cuda:0"` and `recognition.providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]`.

---

## Sample output

See `sample_output/` for:
- `events.log` — real events from a pipeline run
- `face_tracker.db` — SQLite database with registered persons and visits
- `sample_entry.jpg` / `sample_exit.jpg` — real entry/exit face crop snapshots
- `sample_annotated_frame.jpg` — annotated frame with track IDs and entry/exit lines

---

## Assumptions

- SQLite is the sole database (explicit choice under the brief's allowed options).
- ByteTrack is used for tracking (brief allows OpenCV / DeepSORT / ByteTrack).
- Entry/exit geometry uses configurable virtual lines placed at the scene entrance. For camera angles where the frame boundary itself marks entry/exit, place the entry line near the top edge (e.g. `y=0.1`) and the exit line near the bottom (e.g. `y=0.9`).
- InsightFace buffalo_l auto-downloads ~280 MB on first run to `~/.insightface/models/buffalo_l/`.
- A demo video (Loom or YouTube) is required for submission review.

---

## Limitations

- Line-crossing geometry requires manual calibration for the deployment camera angle.
- `min_track_age_frames=3` means a face must appear in at least 3 consecutive detection cycles before ENTRY fires — prevents phantom detections.
- No web dashboard; pipeline prints a console summary.
- CPU inference is ~3–5 seconds per detection frame on a laptop (4K video); use `detection_skip_frames=4` or resize input for real-time performance.

---

## AI usage

AI-assisted development is encouraged by the brief. Prompts used for all major implementation steps are recorded in [AI_PROMPTS.md](AI_PROMPTS.md).

---

## Demo video (submission requirement)

**Demo link (Loom or YouTube):** _TODO — record a 2–3 minute walkthrough showing the pipeline running on the sample video before the submission deadline._

Submit the GitHub repository via the hackathon Google Form before **12:00 PM, Sunday, October 4, 2026**.

---

## Documentation index

| Document | Contents |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System diagrams, module map, SQLite schema |
| [AI_PLANNING.md](AI_PLANNING.md) | Technical decisions, phase plan, edge cases |
| [AI_PROMPTS.md](AI_PROMPTS.md) | Full AI prompt log |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Implementation checklist |
| [sample_output/](sample_output/) | Real run output for submission review |

---

This project is a part of a hackathon run by https://katomaran.com
