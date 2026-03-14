# PROGRESS.md — Build State Tracker

Claude must read this file at the start of every session before touching any code.
Claude must update this file at the end of every session.

---

## Current status

**Phase:** 1 — Core pipeline (single machine, no infrastructure)
**Last worked on:** 2026-03-13 — Implemented pipeline/classifier.py and tests/test_classifier.py; all 33 tests pass
**Next task:** Session 5 — Implement pipeline/embedder.py (CLIP embedding generation)

---

## Completed checkboxes

- [x] Folder structure created
- [x] requirements.txt created
- [x] .env.example created
- [x] ingestion/extractor.py — frame extraction
- [x] pipeline/detector.py — YOLOv8 person detection
- [x] pipeline/classifier.py — color + accessory classification
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

## Files Claude created or modified — Session 4 (2026-03-13)

- Modified: `pipeline/classifier.py` — full implementation (was an empty stub)
- Modified: `tests/test_classifier.py` — 33 tests across three groups (hue mapping, color integration, CLIP classifier)

---

## Files Claude created or modified — Session 3 (2026-03-13)

- Modified: `pipeline/detector.py` — full implementation (was an empty stub)
- Modified: `tests/test_detector.py` — 10 tests covering return type, field types, confidence range, missing file, mocked filtering, multi-person, and bad model name
- Created: `tests/fixtures/sample_frame.jpg` — single JPEG extracted from one.mov for use as a test fixture

---

## Files Claude created or modified — Session 2 (2026-03-13)

- Modified: `ingestion/extractor.py` — full implementation (was an empty stub)
- Created: `tests/test_extractor.py` — 10 tests covering frame count, timestamps, file validity, directory structure, error handling, and configurable FPS

---

## Test results

Session 2 (2026-03-13): 10 passed in tests/test_extractor.py (30.75s)
Session 3 (2026-03-13): 10 passed in tests/test_detector.py (5.39s)
Session 4 (2026-03-13): 33 passed in tests/test_classifier.py (8.82s)

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
