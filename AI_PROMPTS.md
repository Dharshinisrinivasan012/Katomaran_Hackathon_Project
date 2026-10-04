# AI Prompts Log

Structured record of AI-assisted development prompts for this project. Only documented prompts are listed below; future prompts will be appended as work progresses.

---

## Prompt 1

| Field | Value |
|--------|--------|
| **Purpose** | Initial planning phase |
| **Date** | October 2, 2026 |
| **Developer review** | Pending |

### Request summary (faithful)

Planning-only work for hackathon project **"Intelligent Face Tracker with Auto-Registration and Visitor Counting"** in workspace `D:\katomaran`. **Do not implement** face detection, tracking, recognition, or the full application pipeline yet.

**Tasks requested:**

1. Search for Katomaran problem statement PDF or existing files in `D:\katomaran` (and parent paths if needed); use found document as source of truth, otherwise use the detailed requirements in the request.

2. Create planning documentation: `README.md`, `AI_PLANNING.md`, `AI_PROMPTS.md`, `ARCHITECTURE.md`, `config.json`, `requirements.txt`, `.gitignore`, `PROJECT_STATUS.md`.

3. Document in README / ARCHITECTURE / AI_PLANNING: modular folder structure, system architecture, SQLite schema, data flow, entry/exit strategy (exactly one entry + one exit per visit), re-identification strategy, unique visitor counting (no increment on re-recognition), ByteTrack tracking, logging (`events.log` + Python logging), edge cases, CPU/GPU considerations.

4. Optionally create empty placeholder directories with `.gitkeep` (no real detection/recognition implementation).

**Mandatory tech (planning):** Python, YOLO face detection, InsightFace embeddings, ByteTrack or equivalent, OpenCV, SQLite, JSON config, local filesystem for crops, logging. **No** `face_recognition` library. **No** MongoDB/PostgreSQL. Support sample video **and** RTSP (config-driven). `detection_skip_frames` in `config.json`.

**README:** Must include exact hackathon note: *"This project is a part of a hackathon run by https://katomaran.com"*.

**AI_PROMPTS.md:** Prompt #1 only for this phase; do not fabricate other prompts.

### Outcome

Planning documentation and initial project scaffolding (config, dependencies list, gitignore, status checklist, placeholder directories) were generated. Implementation modules remain empty pending developer review.

### Notes

- No Katomaran problem statement PDF was found in the workspace at planning time; requirements above serve as source of truth until a PDF is added.

---

## Prompt 2

| Field | Value |
|--------|--------|
| **Purpose** | Phase 2 foundation — config + SQLite only |
| **Date** | October 2, 2026 |
| **Developer review** | Completed (implementation + pytest) |

### Request summary (faithful)

Phase 1 planning **approved**. Implement **Phase 2 foundation ONLY** — no YOLO, InsightFace, ByteTrack, entry/exit logic, or main video pipeline.

**Documentation:** Search for official Katomaran problem statement PDF; reference in README/AI_PLANNING or note expected path `docs/katomaran-problem-statement.pdf`. Confirm SQLite-only in docs. Update PROJECT_STATUS (developer review + Phase 1 complete; Phase 2 ready → mark items as done). Record this prompt as Prompt #2.

**Config (`src/config/`):** Load `config.json` from project root; optional `CONFIG_PATH` env; validation (required keys, types, sensible ranges e.g. `detection_skip_frames`); clear errors; match existing `config.json` keys.

**Database (`src/database/`):** Clean abstraction; SQLite from config path; init/migrations on first run; schema per ARCHITECTURE.md (persons, face_embeddings, events, visits, optional documented tables); repository CRUD (register person, embeddings, events, open/close visits, gallery list, etc.); parameterized queries; connection separate from repositories.

**Tests:** pytest unit tests for config and database (temp SQLite). pytest in requirements if missing.

**Explicitly out of scope:** detection, tracking, recognition, registration beyond DB fields, entry/exit, pipeline, main video loop.

### Outcome

Implemented `src/config/` (typed loader + validation), `src/database/` (connection, schema init, repositories), and `tests/` with 14 passing pytest cases. Updated README, AI_PLANNING, ARCHITECTURE blurb, PROJECT_STATUS, and PDF placeholder note. No CV pipeline code added.

---

## Prompt 3

| Field | Value |
|--------|--------|
| **Purpose** | Official sample-video source + configurable video input |
| **Date** | October 3, 2026 |
| **Developer review** | Pending |

### Request summary (faithful)

The official Katomaran problem statement provides development sample video through Google Drive folder `https://drive.google.com/drive/folders/15YCN3CYb97GyIoNUV6NJxGNIf-rUBFUJ?usp=sharing`. The developer cannot reliably copy sample video files from Windows Downloads/File Explorer into the Cursor workspace.

**Constraints:** Treat that Drive folder as the official sample video source. Do not fabricate video filenames, paths, or contents. Do not assume videos can be downloaded or accessed automatically from Drive. Design the application so the video input path is configurable. Support a local sample file after later download/copy, and an RTSP URL for the interview. Do not hardcode a specific sample video filename. Add configurable video input to `config.json` (source type `"file"` or `"rtsp"`, video file path, RTSP URL placeholder). If the sample is not physically in the repo, document the official Drive folder and that the local path must be configured after download. Do not commit large sample videos to GitHub unless explicitly required. Keep the official problem statement PDF at `docs/katomaran-problem-statement.pdf`.

**Scope limits:** Do not change the current Phase 2 SQLite/config implementation unless required to support configurable video input. Do not implement YOLO, InsightFace, ByteTrack, entry/exit, or the main video pipeline. Update README.md and AI_PLANNING.md only if necessary to document the official sample-video source and configurable input design. Record this as the next actual prompt in `AI_PROMPTS.md`.

### Outcome

`config.json` `input` now uses `source_type` `"file"` or `"rtsp"`, an empty/configurable `video_path` (no hardcoded sample filename), and an RTSP URL placeholder. Config validation matches those source types. Sample media is gitignored. README and AI_PLANNING document the official Drive folder and post-download path configuration. No CV pipeline or SQLite schema changes.

---

## Prompt 4

| Field | Value |
|--------|--------|
| **Purpose** | Phase 3 — video input + YOLO face detection only |
| **Date** | October 3, 2026 |
| **Developer review** | Pending |

### Exact prompt

```
PHASE 3 — VIDEO INPUT + YOLO FACE DETECTION ONLY

Context:
Phase 1 planning is complete.
Phase 2 SQLite + configuration is implemented and tested.
Prompt 3 added config-driven video input supporting:
- local video file
- RTSP URL

Official sample video source:
https://drive.google.com/drive/folders/15YCN3CYb97GyIoNUV6NJxGNIf-rUBFUJ?usp=sharing

The actual sample video may not yet be present locally.

Implement ONLY the video input and YOLO face detection foundation.

DO NOT implement:
- ByteTrack
- DeepSORT
- InsightFace
- ArcFace
- face embeddings
- face recognition
- face registration
- re-identification
- entry/exit logic
- visitor counting
- database event persistence
- complete application pipeline

### 1. VIDEO INPUT
Create a modular video-input layer under src/input/.
It must support: A. Local video file B. RTSP stream
Use the existing config.json input configuration.
Do not hardcode a sample video filename.
The local video path must come from input.video_path
The RTSP URL must come from input.rtsp_url
Create a clean abstraction so the rest of the application does not care whether frames came from a file or RTSP.
Handle: missing file, invalid path, unable to open video, RTSP connection failure, end of local video, unexpected read failure.
Use clear exceptions and useful logging.

### 2. YOLO FACE DETECTION
Create a separate face-detection module under src/detection/.
Use a YOLO-based face detection model as required by the problem statement.
Do NOT use the Python face_recognition library.
Keep the YOLO implementation behind a clean detector interface/class so the model can be changed later without changing the rest of the application.
The detector should return structured detection results containing at minimum: bounding box, confidence, frame information if needed.
Do not add tracking or recognition yet.

### 3. MODEL CONFIGURATION
Do not silently download or hardcode an arbitrary model.
Add model configuration to config.json in a clean way, including the model path/name and confidence threshold if appropriate.
If a model file is required but is not available yet: document what is required; do not fabricate a model file; do not commit large model binaries to GitHub.

### 4. FRAME PROCESSING
Implement a minimal testable flow: configured video source → OpenCV frame → YOLO detector → face detections.
Respect config.json detection_skip_frames.
The skip-frame behavior must be implemented in a reusable way so it can later be integrated with tracking.

### 5. TEST / DEMO
Create a small development/test entry point that can: open a configured local video if available; read frames; run YOLO detection; draw face bounding boxes; display or optionally save the detection output.
If no local sample video exists: do NOT invent a filename; do NOT fail the entire implementation; provide a clear configuration/error message explaining that input.video_path must be set after downloading the sample video.
Do not build the final application UI.

### 6. TESTS
Add appropriate tests for: video source configuration; invalid file path; source type validation; detector configuration; detection result structure; detection-skip behavior where practical.
Do not write fake tests claiming YOLO successfully detected faces without actually running the model.
If model-dependent tests cannot run because the model is unavailable, clearly mark them as requiring the model rather than fabricating results.

### 7. DOCUMENTATION
Update README.md, AI_PLANNING.md, ARCHITECTURE.md, PROJECT_STATUS.md.
Document: video input abstraction; file vs RTSP configuration; YOLO detector module; detection-skip behavior; model requirements; current limitations.
Do not claim the complete face-tracking system is implemented.

### 8. AI PROMPT RECORD
Add this exact prompt as Prompt #4 in AI_PROMPTS.md.
Record: purpose; exact prompt; files changed; implementation result; tests performed; developer review status.

### 9. DEPENDENCIES
Update requirements.txt only when necessary.
Do not install unnecessary packages.
Before adding a YOLO dependency, verify that it is actually required by the selected implementation.
```

### Files changed

- Created: `src/input/` (`exceptions.py`, `frame.py`, `base.py`, `opencv_source.py`, `factory.py`, `__init__.py`)
- Created: `src/detection/` (`exceptions.py`, `results.py`, `skip.py`, `base.py`, `yolo.py`, `factory.py`, `__init__.py`)
- Created: `src/preview_detections.py`
- Created: `tests/test_input.py`, `tests/test_detection.py`, `tests/test_preview.py`
- Modified: `config.json`, `src/config/loader.py`, `src/config/__init__.py`, `src/__init__.py`, `tests/test_config.py`, `tests/fixtures/invalid_bad_skip_frames.json`, `requirements.txt`, `README.md`, `AI_PLANNING.md`, `ARCHITECTURE.md`, `PROJECT_STATUS.md`, `AI_PROMPTS.md`

### Implementation result

Modular OpenCV `FrameSource` for file and RTSP from `config.json`. Ultralytics `YoloFaceDetector` behind a `FaceDetector` protocol; local weights required, no auto-download. Reusable `DetectionSkipPolicy`. Development preview CLI. SQLite/event pipeline and tracking/recognition not added.

### Tests performed

`python -m pytest tests/ -v` — 34 passed, 1 skipped (`test_yolo_inference_requires_model` skipped because local YOLO weights are not present).

### Developer review status

Pending

---

## Prompt 5

| Field | Value |
|--------|--------|
| **Purpose** | YOLO face-weight identification + first real detection test |
| **Date** | October 3, 2026 |
| **Developer review** | Pending |

### Exact prompt

PHASE 3 — YOLO MODEL SETUP AND REAL DETECTION TEST (local video `data/videos/video_sample1.mp4`; config path `models/yolov8n-face.pt`). Inspect detector and Ultralytics usage; do not download generic COCO / fake / silent arbitrary weights. Locate a local compatible model if present. If source cannot be verified from the project, stop before downloading. Only after the correct model exists: verify weights, run `python -m src.preview_detections --no-display --max-frames 30 --save-dir tmp/preview`, confirm faces, report results. Do not implement tracking/recognition/registration/entry-exit/counting or change SQLite.

### Files changed

- Downloaded (gitignored): `models/yolov8n-face.pt`
- Preview output (not committed): `tmp/preview/frame_000000.jpg` … `frame_000029.jpg`
- Modified: `AI_PROMPTS.md` (this record), `PROJECT_STATUS.md`

### Implementation result

Approved weights downloaded from `https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt` to `models/yolov8n-face.pt` (6,389,512 bytes). `python -m pip install -r requirements.txt` succeeded (`ultralytics 8.4.171`). `YoloFaceDetector` loaded the local file. Preview on `data/videos/video_sample1.mp4` processed 30 frames; detector ran on 10 frames (skip=2) and found faces on **all 10** (8–12 boxes per detect frame; 98 boxes total on detect frames). 30 annotated JPEGs written to `tmp/preview`. No ByteTrack / InsightFace / SQLite / pipeline changes.

### Tests performed

- Ultralytics import + `YoloFaceDetector` load: success
- `python -m src.preview_detections --no-display --max-frames 30 --save-dir tmp/preview`: exit 0; faces detected

### Developer review status

Pending developer review of the live preview outputs in `tmp/preview`.



---

## Prompt 6

| Field | Value |
|--------|--------|
| **Purpose** | ByteTrack verification phase — fix broken tests, run live tracking, document results |
| **Date** | October 3, 2026 |
| **Developer review** | Pending |

### Exact prompt

```
Proceed with the ByteTrack verification phase.
IMPORTANT: Do NOT rebuild or rewrite the existing ByteTrack implementation.
...
[Full prompt as given by developer — six tasks: fix 2 broken tests, run live ByteTrack
integration, verify implementation correctness, update documentation, record prompt #6.]
```

### Files changed

- Modified: `tests/conftest.py` — added `config_with_empty_video_path` fixture (temp config JSON with `video_path=""`)
- Modified: `tests/test_input.py` — `test_create_file_source_requires_configured_path` now uses `config_with_empty_video_path` fixture
- Modified: `tests/test_preview.py` — `test_preview_reports_empty_video_path` now uses `config_with_empty_video_path` fixture
- Modified: `PROJECT_STATUS.md` — phase updated to 4; `src/tracking/` and ByteTrack marked implemented + tested
- Modified: `README.md` — phase line updated; data flow updated; ByteTrack section added; limitations updated
- Modified: `AI_PLANNING.md` — phase table updated; Phase 4 marked current/done
- Modified: `ARCHITECTURE.md` — implemented modules updated; Phase 3–4 diagram and verified results added
- Modified: `AI_PROMPTS.md` — this record (Prompt #6)
- Generated (not committed): `tmp/preview_tracking/frame_000000.jpg` … `frame_000029.jpg`

### Tests performed

- `python -m pytest tests/ -v --tb=short` → **43 passed, 0 failed** (was 41 passed, 2 failed before fixture fix)
- `python -m src.preview_tracking --no-display --max-frames 30 --save-dir tmp/preview_tracking`

### Real tracking test result

| Metric | Value |
|--------|-------|
| Video | `data/videos/video_sample1.mp4` |
| Model | `models/yolov8n-face.pt` (yolov8n-face, WIDER FACE) |
| Frames processed | 30 |
| YOLO detection frames | 10 (skip=2, every 3rd frame) |
| Predicted (skip) frames | 20 |
| Total face detections | 98 |
| Unique track IDs observed | 11 (IDs 1–14, non-contiguous) |
| ID overlap across frame transitions | 29/29 (100%) |
| Preview files written | 30 JPEGs → `tmp/preview_tracking/` |
| Warnings / errors | None |

Key observation: track IDs are **temporary motion identities only** — they are not permanent Face IDs. Permanent Face IDs require InsightFace embeddings and database registration (next phase).

### Developer review status

Pending

---

## Prompt 7

| Field | Value |
|--------|--------|
| **Purpose** | Phase 5 — InsightFace recognition + auto-registration foundation |
| **Date** | October 4, 2026 |
| **Developer review** | Pending |

### Exact prompt

```
Phase 5 — InsightFace Recognition + Auto-Registration
[Full prompt as given — inspect existing codebase first; implement src/recognition/
and src/registration/ using InsightFace buffalo_l and ONNX Runtime; no face_recognition
lib; cosine gallery matching; UUID face_id; SQLite reuse; pytest coverage including
real-model integration tests; verify on video_sample1.mp4; no entry/exit or pipeline yet]
```

### Files created

- `src/recognition/exceptions.py`
- `src/recognition/results.py`        — EmbeddingResult, GalleryMatch, IdentityResult
- `src/recognition/base.py`           — FaceEmbedder, GalleryMatcher protocols
- `src/recognition/similarity.py`     — cosine_similarity, normalize_embedding, CosineMatcher
- `src/recognition/insightface_engine.py` — InsightFaceEngine wrapping buffalo_l ArcFaceONNX
- `src/recognition/factory.py`
- `src/recognition/__init__.py`
- `src/registration/exceptions.py`
- `src/registration/registrar.py`     — FaceRegistrar: gallery load, search, register/re-id
- `src/registration/__init__.py`
- `src/preview_recognition.py`        — YOLO + ByteTrack + InsightFace preview CLI
- `tests/test_recognition.py`         — 29 tests (unit + 3 real-model integration)

### Files modified

- `src/recognition/insightface_engine.py` — added 30% bbox padding for SCRFD face context
- `src/preview_recognition.py`            — updated crop strategy to pass full frame + bbox
- `PROJECT_STATUS.md`, `README.md`, `AI_PLANNING.md`, `ARCHITECTURE.md`, `AI_PROMPTS.md`

### Database changes

None. Existing schema (`persons`, `face_embeddings`, `events`) and all repository
functions were sufficient. No new tables or columns added.

### Config changes

None. `recognition.*` keys already present in config.json were sufficient.

### Tests performed

- `python -m pytest tests/ -v --tb=short` → **72 passed, 0 failed, 0 skipped**
  (43 existing + 29 new recognition tests, including 3 real buffalo_l integration tests)

### Real-model verification result

| Metric | Value |
|--------|-------|
| Video | `data/videos/video_sample1.mp4` |
| InsightFace model | buffalo_l (ArcFaceONNX, 512-dim) |
| Providers | CPUExecutionProvider |
| Frames processed | 30 |
| YOLO detection frames | 10 (skip=2) |
| Total YOLO detections | 98 |
| Embeddings extracted | 70 |
| New face registrations | 10 |
| Re-identifications | 60 |
| Unique persistent face_ids | 10 |
| Cosine similarity range on re-ID | 0.53 – 0.94 |
| Similarity threshold | 0.45 |
| Errors / warnings | None (FutureWarning suppressed in library internals) |
| Preview output | `tmp/preview_recognition/frame_000000.jpg` … `frame_000029.jpg` |

Key note: bbox padding of 30% applied before calling InsightFace SCRFD to ensure
sufficient face context for landmark alignment on 4K video crops.

### Developer review status

Pending

---

## Prompt 8

| Field | Value |
|--------|--------|
| **Purpose** | Phase 6 complete — entry/exit FSM, visit manager, full pipeline, logging, snapshots, sample output, FSM fix, final verification |
| **Date** | October 4, 2026 |
| **Developer review** | Pending |

### Request summary (faithful)

Complete the full hackathon-ready end-to-end Intelligent Face Tracker. Build as much remaining work as possible in one pass. Requirements covered:
- Entry/exit detection (line-crossing FSM, configurable geometry)
- Visit lifecycle (ENTRY/EXIT events, open/close visits in SQLite, total_visits counting)
- events.log (structured JSON-lines, RotatingFileHandler)
- Entry/exit face crop snapshots (data/events/YYYY-MM-DD/)
- Unique visitor counting (persons table, increment only on new ENTRY)
- Complete src/pipeline/main.py end-to-end pipeline entry point
- RTSP readiness (config-driven, existing input abstraction)
- Tests for all new functionality (FSM geometry, crossing, visit lifecycle)
- Real end-to-end run on data/videos/video_sample1.mp4
- Database verification
- Sample output directory for submission
- Documentation updates (README, PROJECT_STATUS, ARCHITECTURE, AI_PLANNING)

### Files created

- `src/pipeline/__init__.py`
- `src/pipeline/visit_manager.py` — VisitManager (ENTRY/EXIT → DB + snapshots)
- `src/pipeline/main.py` — complete end-to-end pipeline entry point
- `tests/test_pipeline.py` — 22 tests (geometry, FSM crossing, visit lifecycle)
- `sample_output/README.md`, `sample_output/db_summary.txt`
- `sample_output/events.log`, `sample_output/face_tracker.db`
- `sample_output/sample_entry.jpg`, `sample_output/sample_exit.jpg`
- `sample_output/sample_annotated_frame.jpg`

### Files modified

- `src/pipeline/entry_exit.py` — added implicit-entry for faces first appearing inside zone
- `tests/test_pipeline.py` — rewrote FSM tests to use fixed geometry independent of config.json
- `config.json` — entry/exit lines calibrated to sample video (y=0.12/0.90), min_track_age=3, debounce=1.0; version → 1.0.0
- `.gitignore` — added sample_output db exclusion
- `README.md` — complete rewrite (clean, no duplicates, correct final state)
- `PROJECT_STATUS.md` — all features marked complete, 94 tests, final DB results
- `AI_PLANNING.md` — header updated to reflect completion
- `ARCHITECTURE.md` — implemented/not-implemented line updated
- `AI_PROMPTS.md` — this record (Prompt #8)
- Deleted: `config_verify.json` (temp file)

### Key implementation decisions

**FSM implicit-entry fix:** Faces in the sample video appear already inside the detection zone (foot_y > entry_line_y) at frame 0. The original FSM only fired on sign transitions (None → nothing, never triggered). Added implicit ENTRY when a face first appears with `entry_sign is None` and `entry_sd > 0` (already inside). This correctly handles both: (a) faces crossing a mid-scene line, and (b) faces appearing in frame from the beginning.

**config.json geometry calibration:** Analysed real face foot_y distribution: range 0.029–0.998, mean 0.339. Moved entry line to y=0.12 (just below typical top-of-frame appearance) and exit line to y=0.90 (near bottom). This produces ENTRY on first confirmed sighting and EXIT when face leaves the frame — matching the brief's "entering/exiting the camera frame" requirement.

**Tests isolated from config.json:** FSM tests use a fixed `EntryExitConfig` (entry y=0.5, exit y=0.7) so they never break when config.json geometry is recalibrated.

### Database changes

None. Existing schema fully sufficient.

### Tests performed

- `python -m pytest tests/ -v --tb=short` → **94 passed, 0 failed**
  (72 existing + 22 new pipeline tests)

### Real end-to-end verification result

| Metric | Value |
|--------|-------|
| Video | `data/videos/video_sample1.mp4` |
| Pipeline command | `python -m src.pipeline.main --no-display --max-frames 30` |
| Frames processed | 30 |
| Detection frames | 10 (skip=2) |
| YOLO detections | 98 |
| Embeddings extracted | 70 |
| New registrations | 10 |
| Re-identifications | 60 |
| ENTRY events | 8 |
| EXIT events | 8 |
| Closed visits | 8 |
| Open visits remaining | 0 |
| Unique visitors (DB) | 10 |
| Event snapshots | 8 entry + 8 exit JPEGs in `data/events/2026-10-04/` |
| events.log | REGISTER/RECOGNIZE/ENTRY/EXIT/SYSTEM JSON-lines |
| DB consistency | All ENTRY/EXIT pairs matched, no orphaned visits |

### Developer review status

Pending

---

## Prompt 9

| Field | Value |
|--------|--------|
| **Purpose** | Final submission audit — fix stale documentation, verify all hackathon requirements, final git commit |
| **Date** | October 4, 2026 |
| **Developer review** | Pending |

### Request summary (faithful)

Final submission stage. Inspect the entire repository. Implement/finish all remaining submission-critical work in one pass: audit all docs, fix stale numbers, verify RTSP readiness, harden error handling, run tests (must keep 94 passing), run final clean demo, git cleanup and final commit.

### Changes made

- `README.md` — fixed tech stack test count (93→94), updated config table to actual values (entry/exit y=0.12/0.90, min_track_age=3, debounce=1.0), fixed sample summary numbers (8 ENTRY+EXIT, 10 visitors), improved demo command, clarified demo video instructions
- `ARCHITECTURE.md` — updated verified results table to actual 30-frame run numbers; replaced "frame boundary mode planned" with accurate current implementation description
- `.gitignore` — restored sample_output images (removed over-aggressive *.jpg exclusion); added config_*.json exclusion
- `sample_output/README.md` — updated with correct verified metrics
- `sample_output/db_summary.txt` — updated with actual run data
- `sample_output/sample_entry.jpg`, `sample_entry.jpg`, `sample_annotated_frame.jpg` — refreshed from latest run
- `src/pipeline/main.py` — added `--video` and `--db-path` CLI overrides
- `src/pipeline/__main__.py` — created for `python -m src.pipeline` shorthand
- `requirements.txt` — cleaned up (removed unused lapx/filterpy/scipy/PyYAML)
- `PROJECT_STATUS.md` — rewritten clean with correct numbers

### Tests performed

`python -m pytest tests/ -v --tb=short` → **94 passed, 0 failed**

### Final demo result (isolated :memory: DB)

| Metric | Value |
|--------|-------|
| Video | `data/videos/video_sample1.mp4` |
| Frames | 30 (10 detection, 20 predicted) |
| YOLO detections | 98 |
| Embeddings | 70 |
| New registrations | 10 |
| Re-identifications | 60 |
| ENTRY events | 8 |
| EXIT events | 8 |
| Closed visits | 8 |
| Open visits | 0 |
| Unique visitors | 10 |
| Annotated frames | 30 (tmp/final_demo/) |
| Event snapshots | 8+8 JPEGs (data/events/2026-10-04/) |

### Git commits

- `dcf26e0` — Complete end-to-end intelligent face tracker (Phases 1–6)
- `003d301` — Finalize hackathon submission

### Developer review status

Pending
