# AI Planning — Technical Plan (Phase 1)

**Project:** Intelligent Face Tracker with Auto-Registration and Visitor Counting  
**Workspace:** `D:\katomaran`  
**Phase:** Planning only — no CV pipeline implementation in this phase.

**Source of truth:** [docs/katomaran-problem-statement.pdf](docs/katomaran-problem-statement.pdf) (Katomaran hackathon brief). Supplementary notes in this file and [ARCHITECTURE.md](ARCHITECTURE.md) must not add mandatory requirements beyond the PDF unless labeled as an explicit design choice.

**Persistence:** SQLite only (`database.sqlite_path` in config) — no MongoDB or PostgreSQL.

---

## 1. Objectives

| Objective | Success criterion |
|-----------|-------------------|
| Detect faces | Stable bounding boxes on video and RTSP |
| Track individuals | Consistent `track_id` via ByteTrack between detection frames |
| Recognize / register | Persistent `face_id` via InsightFace embeddings |
| Count unique visitors | Increment visit count only on new visit ENTRY, not on re-recognition |
| Audit trail | SQLite events + `logs/events.log` |
| Config-driven | All thresholds and paths in `config.json` |

---

## 2. Proposed folder structure

```
src/
  input/
    # VideoCapture wrapper, RTSP reconnect, frame iterator protocol
  detection/
    # YOLO load/infer; respect detection_skip_frames
  tracking/
    # ByteTrack adapter; Detection → Track list
  recognition/
    # Face align, embed, gallery search (no face_recognition lib)
  registration/
    # Quality gates, face_id issuance, crop save, DB insert
  database/
    # Schema init, repositories (persons, embeddings, events, visits)
  logging/
    # setup_logging(), EventLogger → events.log
  config/
    # load_config(), validation, typed accessors
  pipeline/
    # main loop, wiring, graceful shutdown
tests/
  unit/
  integration/
data/
  samples/
  crops/
  faces/
  events/
  sqlite/
logs/
models/
```

**Convention:** Each package may contain empty `__init__.py` only until implementation. No YOLO/InsightFace logic in Phase 1.

---

## 3. Implementation phases (recommended)

| Phase | Scope |
|-------|--------|
| **1 (current)** | Docs, config, schema, placeholders |
| **2 (in progress)** | Config loader + validation, SQLite init/repositories, unit tests (logging + input deferred to later Phase 2 items) |
| **3** | Detection + ByteTrack on video; visualize tracks |
| **4** | InsightFace embed + gallery + auto-registration |
| **5** | Entry/exit FSM, visits, unique counting, events.log |
| **6** | Tests, tuning, CPU/GPU profiles, demo script |

---

## 4. Pipeline algorithm (pseudocode)

```
config = load_config("config.json")
init_db(config.database.sqlite_path)
init_logging(config.logging)

source = create_input(config.input)
detector = FaceDetector(config.detection)      # Phase 3+
tracker = ByteTracker(config.tracking)       # Phase 3+
recognizer = InsightFaceEngine(config.recognition)  # Phase 4+
registrar = AutoRegistrar(config.registration)       # Phase 4+
visit_logic = EntryExitManager(config.entry_exit)    # Phase 5+

frame_idx = 0
last_detections = []

for frame in source:
    frame_idx += 1
    run_det = (frame_idx % (config.detection.detection_skip_frames + 1) == 1)

    if run_det:
        last_detections = detector.detect(frame)
    tracks = tracker.update(frame, last_detections)

    for track in tracks eligible for recognition:
        face_id = track.face_id or recognizer.identify(track.crop)
        if face_id is None and registrar.should_register(track):
            face_id = registrar.register(track, frame)
        track.face_id = face_id

        visit_logic.process(track, face_id, timestamp)

    optionally render / publish metrics
```

All numeric parameters from config — never literals in module code.

---

## 5. Data flow (detailed)

1. **Input** decodes BGR frames; RTSP path uses reconnect backoff from config.
2. **Detection** (every N frames) produces face boxes; intermediate frames rely on tracker prediction.
3. **Tracking** outputs `track_id`, bbox, age, state (tentative/confirmed/lost).
4. **Recognition** runs on confirmed tracks at a configurable interval (future key: `recognition.interval_frames`) to limit InsightFace cost.
5. **Registration** creates UUID `face_id`, writes crop to `data/faces/{face_id}/`, inserts `persons` + `face_embeddings`, logs REGISTER event.
6. **Entry/exit** consumes track position history; writes ENTRY/EXIT to `events` and mutates `visits`.
7. **Database** commits in short transactions to avoid locking on edge devices.

---

## 6. Entry / exit and visit counting

### Rules

1. A **visit** begins with the first valid **ENTRY** event for a `face_id` while no `open` visit exists for that person.
2. A **visit** ends with **EXIT** for the same `face_id` that closes the open visit.
3. **Exactly one ENTRY and one EXIT** per visit row when `status=closed`.
4. **`persons.total_visits`** increments once when a new open visit is created (at ENTRY), **not** when:
   - the same person is recognized on every frame;
   - embedding match finds a returning person before ENTRY;
   - a person re-enters after a completed visit (that is a **new** visit and increments again — intentional for visit counting).

### Re-recognition vs unique visitors

| Action | Affects `total_visits`? | Affects daily unique set? |
|--------|-------------------------|---------------------------|
| RECOGNIZE (known face in frame) | No | No (already counted for open visit) |
| REGISTER (new face_id) | No until ENTRY | No until ENTRY |
| ENTRY (new open visit) | Yes (+1) | Yes (add to day's unique set) |
| EXIT | No | No |

**Daily unique visitors:** Derive from `visits` where `entry_at` date = today, or maintain `daily_stats` updated on ENTRY.

### Entry/exit geometry

**Brief requirement:** Each face entering and leaving the **camera frame** must produce exactly one entry and one exit log (crop, timestamp, event type, Face ID).

**Planned implementation:** Configurable `entry_exit.mode` (see `config.json`). Default **`line_crossing`** uses virtual lines for directional counting at an entrance scene; **`frame_boundary`** (Phase 5+) will treat crossing the frame edge as entry/exit to match the brief literally. Lines can also be placed at frame borders to approximate boundary detection.

- Direction vectors from config prevent false triggers when walking parallel to a line.
- **Debounce:** Minimum time outside/inside between ENTRY and EXIT to handle straddling.

### Open visit timeout (edge case)

If EXIT is never seen (camera blind spot), close visit after `visit_timeout_sec` (future config key) with SYSTEM event — document only for Phase 2+.

---

## 7. Re-identification strategy

1. **Embedding extraction:** InsightFace `buffalo_l` (or config `model_name`) → L2-normalized vector.
2. **Gallery search:** Brute-force cosine similarity against all embeddings or per-person centroid (start simple: all embeddings, take max per `face_id`).
3. **Match:** If `sim >= recognition.similarity_threshold`, assign that `face_id`.
4. **Reject:** Below threshold → unknown; registration path if enabled and track quality passes.
5. **Track binding:** Cache `track_id → face_id` for confirmed matches; re-verify every M frames to recover from ID switches.
6. **ID switch handling:** If new track gets high sim to known face while another track same face active, prefer older track or higher track age — log WARNING.

**No `face_recognition` library** — only InsightFace + numpy cosine similarity.

---

## 8. Tracking strategy (ByteTrack)

- **Why ByteTrack:** Strong with low-FPS detection thanks to Kalman prediction; fits `detection_skip_frames > 0`.
- **Inputs:** Face boxes from YOLO (class-agnostic face model).
- **Parameters:** `track_thresh`, `match_thresh`, `track_buffer` from `config.json`.
- **Lost tracks:** After `track_buffer` frames without match, emit TRACK_LOST; entry/exit state for that track_id frozen; do not count EXIT unless person was Inside.

---

## 9. Logging

| Sink | Purpose |
|------|---------|
| Python `logging` | DEBUG/INFO for developers; module names |
| `logs/events.log` | One line per business event (JSON recommended in implementation) |

Suggested event log fields: `timestamp`, `event_type`, `face_id`, `track_id`, `visit_id`, `message`, `extra`.

Rotate via `logging.handlers.RotatingFileHandler` using config sizes.

**Privacy:** Do not log raw embeddings when `log_embeddings` is false (default).

---

## 10. Edge cases

| Case | Mitigation |
|------|------------|
| Missed detection frames | ByteTrack prediction; lower `detection_skip_frames` if IDs fragment |
| Occlusion | Short buffer; re-associate on reappear via embedding |
| Track ID switch | Periodic re-id; merge visits only same `face_id`, not track_id |
| Twin / similar faces | Raise threshold; require multiple frame consensus before REGISTER |
| Partial face / blur | `quality_min_score`, `min_face_size_px` gates |
| Duplicate registration | Check gallery before insert; embedding_hash UNIQUE |
| Person loiters on line | Direction + debounce + min track age |
| Two people one box | Rare at entrance if camera angled; prefer min box width split (future) |
| RTSP drop | Reconnect with delay; log SYSTEM; do not corrupt open visits |
| Clock skew | Use monotonic time for debounce; UTC ISO strings in DB |
| Same person re-entry same day | New visit after EXIT → increment total_visits (by design) |
| Person never exits | Visit timeout closes open visit (Phase 2+ config) |

---

## 11. CPU and GPU considerations

### CPU-only (default hackathon laptop)

- YOLO nano face model, `detection_skip_frames`: 2–4.
- InsightFace with `CPUExecutionProvider`.
- Resize input (`detection.input_size` 640 or 480).
- Limit recognition to every 3–5 frames per track.
- SQLite WAL mode enabled in config.

### GPU acceleration

- Install `onnxruntime-gpu` and CUDA-compatible PyTorch for Ultralytics if using GPU inference.
- Set `detection.device`: `"cuda:0"` and `recognition.providers`: `["CUDAExecutionProvider", "CPUExecutionProvider"]`.
- Watch VRAM: batch size 1 for real-time stream.
- `performance.use_half_precision`: true when supported.

### Profiling checklist (Phase 6)

- Measure ms/frame: capture, detect, track, recognize.
- Tune skip frames until stable 10–15 FPS minimum on target hardware.

---

## 12. Testing strategy (future)

- **Unit:** config validation, line-crossing math, cosine match, visit FSM transitions.
- **Integration:** Short sample video with scripted ENTRY/EXIT; assert DB row counts.
- **Regression:** Frozen embeddings for two test crops → stable face_id match.

---

## 13. Dependencies rationale

| Package | Role |
|---------|------|
| opencv-python | Video/RTSP, crop, draw |
| ultralytics | YOLO inference |
| insightface | Detection/recognition models (face) |
| onnxruntime | Embedding inference backend |
| lapx / filterpy / scipy | ByteTrack assignment & Kalman |

---

## 14. Open decisions

1. Exact YOLO face weight file (download URL, license).
2. Whether to store one embedding per person or multiple (schema supports multiple).
3. CLI vs minimal web dashboard for demo.
4. Align entry/exit crop directory with brief example (`logs/entries/YYYY-MM-DD/`) vs current `paths.events_snapshot_dir` at demo time if reviewers expect that layout.

---

## 15. Acceptance criteria for Phase 2 kickoff

- [ ] Developer sign-off on this plan  
- [ ] Sample video placed in `data/samples/`  
- [ ] Models downloaded to `models/`  
- [ ] `config.json` calibrated for deployment camera geometry  

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for living checklist.
