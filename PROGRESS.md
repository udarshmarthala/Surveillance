# PROGRESS.md — Build State Tracker

Claude must read this file at the start of every session before touching any code.
Claude must update this file at the end of every session.

---

## Current status

**Phase:** 1 — Core pipeline (single machine, no infrastructure)
**Last worked on:** 2026-03-13 — Created full folder structure, empty stub files, requirements.txt, .env.example, .gitignore; updated CLAUDE.md to reference DECISIONS.md and PROGRESS.md
**Next task:** Session 2 — Implement ingestion/extractor.py (frame extraction with OpenCV)

---

## Completed checkboxes

- [x] Folder structure created
- [x] requirements.txt created
- [x] .env.example created
- [ ] ingestion/extractor.py — frame extraction
- [ ] pipeline/detector.py — YOLOv8 person detection
- [ ] pipeline/classifier.py — color + accessory classification
- [ ] pipeline/embedder.py — CLIP embedding generation
- [ ] pipeline/processor.py — full pipeline orchestrator
- [ ] storage/models.py — data models
- [ ] storage/db.py — SQLite storage
- [ ] storage/vector_store.py — ChromaDB storage
- [ ] scripts/setup_db.py — database initializer
- [ ] scripts/ingest_video.py — manual pipeline runner
- [ ] search/query_parser.py — text query parser
- [ ] search/searcher.py — hybrid SQL + ANN search
- [ ] search/api.py — FastAPI endpoint
- [ ] frontend — React search UI

---

## What is currently broken or incomplete

[Claude fills this in after each session. Be specific — file name, function name, what the error was.]

Nothing yet.

---

## Test results

[Claude fills this in after running pytest each session.]

Nothing yet.

---

## Files Claude created or modified this session

**Session 1 (2026-03-13):**
- Created: all directories under project root
- Created: `ingestion/extractor.py`, `ingestion/watcher.py`
- Created: `pipeline/detector.py`, `pipeline/classifier.py`, `pipeline/embedder.py`, `pipeline/processor.py`
- Created: `storage/db.py`, `storage/vector_store.py`, `storage/models.py`
- Created: `search/query_parser.py`, `search/searcher.py`, `search/api.py`
- Created: `frontend/src/App.jsx`, `frontend/src/components/SearchBar.jsx`, `ResultGrid.jsx`, `VideoPlayer.jsx`, `CameraMap.jsx`, `package.json`
- Created: `tests/test_detector.py`, `test_classifier.py`, `test_searcher.py`
- Created: `scripts/ingest_video.py`, `setup_db.py`, `benchmark.py`
- Created: `docker/Dockerfile.pipeline`, `Dockerfile.api`, `docker-compose.yml`
- Created: `requirements.txt`, `.env.example`, `.gitignore`
- Created: `__init__.py` in `ingestion/`, `pipeline/`, `storage/`, `search/`, `tests/`
- Modified: `CLAUDE.md` — added companion files section referencing DECISIONS.md and PROGRESS.md
- Modified: `PROGRESS.md` — updated session state

---

## Errors we hit and how we fixed them

[Claude logs every significant error and its fix here. This prevents hitting the same error twice.]

Nothing yet.
