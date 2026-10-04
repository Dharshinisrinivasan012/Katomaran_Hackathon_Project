# Sample Output — Intelligent Face Tracker

Generated from: `data/videos/video_sample1.mp4`  
Run command: `python -m src.pipeline.main --no-display --max-frames 30 --save-dir tmp/final_demo --db-path :memory:`  
Date: 2026-10-04

## Contents

| File | Description |
|------|-------------|
| `events.log` | Full structured audit log (JSON-lines per event) |
| `db_summary.txt` | Human-readable DB summary with consistency checks |
| `sample_entry.jpg` | Real face crop saved at ENTRY event |
| `sample_exit.jpg` | Real face crop saved at EXIT event |
| `sample_annotated_frame.jpg` | Annotated frame with track IDs, face IDs, entry/exit lines |

## Verified pipeline result (30 frames, isolated :memory: DB)

| Metric | Value |
|--------|-------|
| Frames processed | 30 |
| YOLO detection frames | 10 (skip=2, every 3rd frame) |
| Total YOLO detections | 98 |
| Embeddings extracted | 70 |
| New face registrations | 10 |
| Re-identifications | 60 |
| ENTRY events | 8 |
| EXIT events | 8 |
| Closed visits | 8 |
| Open visits remaining | 0 |
| Unique visitors | 10 |

Entry/exit snapshots saved to: `data/events/2026-10-04/` (8 entry + 8 exit JPEGs)

## Sample events.log lines

```json
{"event":"REGISTER","face_id":"FT-8ad361db...","track_id":1,"frame":0,"ts":"2026-10-04T..."}
{"event":"RECOGNIZE","face_id":"FT-8ad361db...","track_id":1,"sim":0.8954,"frame":3,"ts":"..."}
{"event":"ENTRY","face_id":"FT-8ad361db...","track_id":1,"visit_id":1,"frame":2,"snapshot":"data/events/2026-10-04/entry_8ad361db.jpg","ts":"..."}
{"event":"EXIT","face_id":"FT-8ad361db...","track_id":1,"visit_id":1,"frame":61,"snapshot":"data/events/2026-10-04/exit_8ad361db.jpg","ts":"..."}
{"event":"SYSTEM","message":"pipeline finished","frames":30,"unique_visitors":10,"entries":8,"exits":8,"ts":"..."}
```
