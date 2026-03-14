# CLAUDE.md — Surveillance Footage Search System

This file tells Claude Code how to work in this project. Read it fully before doing anything.

---

## Companion files — read these every session

This project uses three tracking files that work together. Claude must read all three at the start of every session, in this order:

| File | Purpose | When Claude updates it |
|---|---|---|
| `CLAUDE.md` | Conventions, architecture, rules — the rulebook | When a new convention is established or a rule changes |
| `DECISIONS.md` | Architectural decisions and what NOT to do | When a significant new architectural choice is made during the session |
| `PROGRESS.md` | Current build state, what is done, what is broken | At the **end of every session**, always |
| `SCHEMA.md` | DB column names, types, enum values, query parser output shape — read-only reference | Only when a schema decision changes; update here first, then log in DECISIONS.md, then update code |
| `API_CONTRACT.md` | FastAPI endpoint URLs, request/response shapes — read-only reference | Only when an endpoint contract changes; update here first, then update `api.py` and frontend together |

### How Claude should use these files

- **Before touching any code:** read `PROGRESS.md` to know exactly where work left off, then read `DECISIONS.md` to check if your intended approach has already been decided.
- **During the session:** if you make a new architectural choice (picking a library, choosing an approach, ruling something out), log it in `DECISIONS.md` immediately in the existing format.
- **At the end of every session:** update `PROGRESS.md` — mark completed checkboxes, log the "last worked on" date, set the next task, record any errors hit and how they were fixed, and list every file created or modified.

### These files are living documents — iterate them freely

Claude is expected to update and evolve all three files as the project grows. Specifically:

- If a convention in `CLAUDE.md` turns out to be wrong or needs refinement after real implementation, update it here — don't silently ignore it.
- If a decision in `DECISIONS.md` needs to be revisited because of a real technical blocker, add a note under the existing decision explaining the new context before changing anything.
- If `PROGRESS.md` gains new items as scope expands, add them to the checklist. If a task splits into subtasks mid-build, break it up.
- Never let these files go stale. An outdated tracking file is worse than no tracking file.

---

## What this project is

A system that searches surveillance video footage to find a person based on clothing color and accessories (e.g. "red shirt, glasses"). It processes video files, detects people in frames, extracts visual attributes, stores them, and lets users search with natural language or attribute filters.

---

## Project structure

```
surveillance-search/
├── CLAUDE.md                  # This file
├── README.md
├── requirements.txt
├── .env                       # API keys and config — never commit this
├── .env.example               # Safe template to commit
│
├── ingestion/                 # Video input and frame extraction
│   ├── extractor.py           # OpenCV frame extraction from video files
│   └── watcher.py             # Watch a folder for new video files
│
├── pipeline/                  # Core CV inference pipeline
│   ├── detector.py            # YOLOv8 person detection
│   ├── classifier.py          # Clothing attribute classification (color, type)
│   ├── embedder.py            # CLIP embedding generation per person crop
│   └── processor.py           # Orchestrates detector → classifier → embedder
│
├── storage/                   # Database layer
│   ├── db.py                  # PostgreSQL connection and queries (SQLite for dev)
│   ├── vector_store.py        # Milvus / Qdrant vector DB interface
│   ├── models.py              # Data models / schema definitions
│   └── migrations/            # SQL migration files
│
├── search/                    # Search and query logic
│   ├── query_parser.py        # Parse text query into structured filters + embedding
│   ├── searcher.py            # Combines SQL filter + ANN vector search + re-ranking
│   └── api.py                 # FastAPI routes for search endpoints
│
├── frontend/                  # React web UI
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── SearchBar.jsx
│   │   │   ├── ResultGrid.jsx
│   │   │   ├── VideoPlayer.jsx
│   │   │   └── CameraMap.jsx
│   └── package.json
│
├── tests/                     # All tests go here
│   ├── test_detector.py
│   ├── test_classifier.py
│   ├── test_searcher.py
│   └── fixtures/              # Sample video clips and images for testing
│
├── scripts/                   # One-off utility scripts
│   ├── ingest_video.py        # Manually run pipeline on a single video file
│   ├── setup_db.py            # Initialize database tables and vector collections
│   └── benchmark.py           # Test inference speed on sample data
│
└── docker/                    # Docker config (later stage)
    ├── Dockerfile.pipeline
    ├── Dockerfile.api
    └── docker-compose.yml
```

---

## Tech stack

| Layer | Technology | Notes |
|---|---|---|
| Language | Python 3.11 | All backend code |
| Frame extraction | OpenCV (`cv2`) | Extract keyframes at 2fps |
| Person detection | YOLOv8 via `ultralytics` | Use `yolov8n.pt` for dev speed, `yolov8m.pt` for accuracy |
| Clothing classifier | CLIP via `transformers` (HuggingFace) | Zero-shot color + attribute classification |
| Embedding model | `openai/clip-vit-base-patch32` | 512-dim embeddings per person crop |
| Metadata DB | SQLite (dev) → PostgreSQL (prod) | Structured attributes: color, timestamp, camera, bbox |
| Vector DB | ChromaDB (dev) → Milvus (prod) | ANN search on CLIP embeddings |
| API | FastAPI | Search endpoint + video serving |
| Frontend | React + Vite | Simple search UI |
| Task queue | Celery + Redis | Async video processing (add this after core works) |

**Dev-first rule:** Always use SQLite + ChromaDB locally. Only switch to PostgreSQL + Milvus when deploying. Do not add infrastructure complexity before the CV pipeline is working.

---

## Environment variables

All secrets and config go in `.env`. Never hardcode them. The `.env.example` shows all required keys without values.

```
# .env.example
DATABASE_URL=sqlite:///./dev.db
VECTOR_DB_PATH=./chroma_db
VIDEO_STORAGE_PATH=./data/videos
FRAME_STORAGE_PATH=./data/frames
KEYFRAME_FPS=2
YOLO_MODEL=yolov8n.pt
CLIP_MODEL=openai/clip-vit-base-patch32
CONFIDENCE_THRESHOLD=0.5
```

---

## How the pipeline works (data flow)

```
Video file
    ↓
extractor.py       — reads video, pulls 1 frame every 0.5s, saves as JPEG
    ↓
detector.py        — runs YOLOv8 on each frame, returns bounding boxes for persons only
    ↓
classifier.py      — crops each detected person, runs CLIP zero-shot to get:
                       - dominant color (red/blue/black/white/green/yellow/grey/brown)
                       - garment type (shirt/jacket/dress/shorts/pants)
                       - accessories (glasses/hat/bag/none)
    ↓
embedder.py        — generates 512-dim CLIP embedding of each person crop
    ↓
db.py              — stores: frame_id, camera_id, timestamp, bbox, color, garment, accessories
vector_store.py    — stores: embedding + reference to the detection record ID
```

**Search flow:**
```
User query: "red shirt, glasses"
    ↓
query_parser.py    — extracts: {color: "red", accessories: ["glasses"]}
                     also generates CLIP text embedding of the full query string
    ↓
searcher.py        — runs SQL filter on color + accessories columns (hard filter)
                     runs ANN search on vector DB using text embedding (soft match)
                     re-ranks results: score = 0.6 * vector_score + 0.4 * attribute_score
    ↓
api.py             — returns top N results with: thumbnail path, timestamp, camera_id, score
```

---

## Coding conventions

**General:**
- All Python files use type hints. No untyped function signatures.
- Every function has a one-line docstring explaining what it does, not how.
- Max function length: 40 lines. If longer, split it.
- No global state. Pass config explicitly via function arguments or a Config dataclass.
- All file paths go through `pathlib.Path`, never raw strings.

**Error handling:**
- Never use bare `except:`. Always catch specific exceptions.
- If a video frame fails to process, log the error and skip the frame. Do not crash the whole pipeline.
- If a model fails to load, raise immediately with a clear message — don't silently fail.

**Logging:**
- Use Python's `logging` module everywhere. No `print()` statements in production code.
- Log at INFO level: pipeline start/end, frame counts, detection counts.
- Log at DEBUG level: individual detection results, embedding shapes.
- Log at ERROR level: anything that causes a frame or video to be skipped.

**Models:**
- Models are loaded once at startup, not per-frame. Load in `__init__` or a dedicated `load_model()` function.
- Always run inference in a `torch.no_grad()` context.
- YOLOv8 confidence threshold default is 0.5 — make it configurable via env var.

**Database:**
- Use parameterized queries only. No f-string SQL — ever.
- All DB writes happen in transactions. Roll back on failure.
- Schema changes go in numbered migration files under `storage/migrations/`.

**Color detection:**
- Convert BGR (OpenCV default) to HSL before classifying color.
- Map hue ranges to color names: 0–15° and 345–360° = red, 15–45° = orange, 45–75° = yellow, 75–150° = green, 150–255° = blue, 255–315° = purple, 315–345° = pink.
- Store as a string enum in the DB, not free text.

---

## Common tasks

**Run the pipeline on a single video file:**
```bash
python scripts/ingest_video.py --video path/to/video.mp4 --camera-id cam_01
```

**Initialize the database:**
```bash
python scripts/setup_db.py
```

**Start the search API:**
```bash
uvicorn search.api:app --reload --port 8000
```

**Start the frontend:**
```bash
cd frontend && npm run dev
```

**Run all tests:**
```bash
pytest tests/ -v
```

**Check pipeline speed on sample data:**
```bash
python scripts/benchmark.py --video tests/fixtures/sample.mp4
```

---

## Current status and what to build next

Progress is tracked in `PROGRESS.md` — do not duplicate it here. Read that file for the current build state, completed items, and what comes next.

---

## What Claude should NOT do in this project

- Do not add Kafka, Kubernetes, or any distributed infrastructure until the core pipeline works end-to-end on a single machine.
- Do not use async/await in the pipeline layer — keep it synchronous until Celery is added.
- Do not store raw video frames in the database. Store file paths only.
- Do not load models inside a loop. Models load once, inference runs many times.
- Do not use face recognition — this project identifies by clothing and accessories only.
- Do not write frontend code unless the API endpoint it depends on already exists and is tested.

---

## Dependencies

Install everything with:
```bash
pip install -r requirements.txt
```

Key packages:
```
ultralytics>=8.0.0        # YOLOv8
transformers>=4.35.0      # CLIP via HuggingFace
torch>=2.0.0              # PyTorch (CPU is fine for dev)
opencv-python>=4.8.0      # Frame extraction and image processing
chromadb>=0.4.0           # Vector DB for development
fastapi>=0.104.0          # Search API
uvicorn>=0.24.0           # ASGI server
sqlalchemy>=2.0.0         # ORM for SQLite/PostgreSQL
python-dotenv>=1.0.0      # .env loading
pillow>=10.0.0            # Image handling
numpy>=1.24.0             # Array operations
pytest>=7.4.0             # Testing
```

---

## Notes for Claude

- This project is being built by a beginner. Explain what you're doing and why, not just how.
- When creating a new file, always check this CLAUDE.md first to make sure the file belongs in the right place and follows conventions.
- When asked to implement a feature, implement the simplest working version first, then ask if optimization is needed.
- When something could be done two ways, pick the simpler one unless there's a strong reason not to.
- Always write a test for any new function in the pipeline or search layer.
