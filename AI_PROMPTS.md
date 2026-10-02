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
