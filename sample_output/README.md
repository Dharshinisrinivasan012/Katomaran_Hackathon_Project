# Sample Output — Intelligent Face Tracker

Generated from: `data/videos/video_sample1.mp4`  
Run command: `python -m src.pipeline.main --no-display --max-frames 30 --save-dir tmp/final_run`  
Date: 2026-10-04  
Config: `entry_line y=0.12`, `exit_line y=0.90`, `min_track_age_frames=3`, `debounce_sec=1.0`

## Contents

| File | Description |
|------|-------------|
| `events.log` | Full structured audit log (JSON-lines per event) |
| `face_tracker.db` | SQLite database with persons, embeddings, events, visits |
| `db_summary.txt` | Human-readable DB summary with consistency checks |
| `sample_entry.jpg` | Real face crop saved at ENTRY event |
| `sample_exit.jpg` | Real face crop saved at EXIT event |
| `sample_annotated_frame.jpg` | Annotated frame with track IDs, face IDs, entry/exit lines |

## Key results (30-frame pipeline run, fresh DB)

| Metric | Value |
|--------|-------|
| Frames processed | 30 |
| YOLO detection frames | 10 (skip=2) |
| Total YOLO detections | 98 |
| Embeddings extracted | 70 |
| New registrations | 10 |
| Re-identifications | 60 |
| ENTRY events | 8 |
| EXIT events | 8 |
| Closed visits | 8 |
| Unique visitors (DB persons) | 10 |
| Open visits remaining | 0 |

Entry/exit snapshots saved to: `data/events/2026-10-04/` (8 entry + 8 exit JPEGs)
