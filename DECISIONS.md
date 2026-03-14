# DECISIONS.md — Architectural Decisions Log

Claude must read this file before suggesting any changes to libraries, 
tools, or architecture. If a decision is already logged here, do not 
re-open it unless there is a critical technical reason — and if so, 
explain why before changing anything.

---

## Decision 1: SQLite for development, not PostgreSQL

**Decision:** Use SQLite locally during all development. Switch to PostgreSQL only when deploying.

**Why:** PostgreSQL requires a running server, Docker, and configuration. SQLite is a single file. The beginner building this project should not be debugging database connections while also learning computer vision. SQLAlchemy abstracts the difference — swapping later is one line change in the connection string.

**Do not suggest switching to PostgreSQL until the full pipeline works end-to-end.**

---

## Decision 2: ChromaDB for vector storage in dev, not Milvus

**Decision:** Use ChromaDB locally. Switch to Milvus or Qdrant for production.

**Why:** ChromaDB runs in-process with zero setup. Milvus requires Docker and a running server. Same reason as Decision 1 — don't add infrastructure before the CV code works.

**Do not suggest Milvus, Qdrant, Pinecone, or Weaviate until the search layer is fully working.**

---

## Decision 3: CLIP for both classification and embedding, not separate models

**Decision:** Use CLIP (openai/clip-vit-base-patch32) for zero-shot clothing attribute classification AND for generating person embeddings. Do not add a separate fine-tuned clothing classifier model.

**Why:** Adding a second model (like a fine-tuned EfficientNet) requires a training dataset, GPU memory management for two models, and more complexity. CLIP zero-shot is good enough for a working prototype and uses one model for two tasks.

**Do not suggest adding a separate clothing classifier model until CLIP accuracy is proven insufficient on real test videos.**

---

## Decision 4: Synchronous pipeline, not async

**Decision:** The pipeline runs synchronously. No asyncio, no threading in the core pipeline code.

**Why:** The person building this is a beginner. Async bugs are hard to debug. Celery will be added later for background processing, but the pipeline functions themselves stay synchronous and simple.

**Do not add async/await to pipeline/, ingestion/, or storage/ code.**

---

## Decision 5: No face recognition

**Decision:** This system identifies people by clothing and accessories only. No face detection, no face recognition, no face embeddings.

**Why:** Privacy, legal complexity, and scope. The search feature is "red shirt, glasses" — not "find this specific person." Keeping it attribute-based keeps it simpler and avoids significant ethical and legal issues.

**Do not add any face recognition library or face-based identification feature.**

---

## Decision 6: 2fps keyframe extraction

**Decision:** Extract one frame every 0.5 seconds (2fps). Do not extract every frame.

**Why:** A 1-hour video at 30fps = 108,000 frames. At 2fps = 7,200 frames. Processing 108k frames per video on a laptop is not feasible. 2fps catches any person who appears for more than half a second, which is sufficient for surveillance use cases.

**Do not change this without benchmarking the storage and inference cost first.**

---

## Decision 7: YOLOv8n for development

**Decision:** Use yolov8n.pt (nano, fastest) during development. Switch to yolov8m.pt (medium) for production accuracy testing.

**Why:** On a laptop CPU, yolov8n runs at ~15fps on 640px images. yolov8m runs at ~5fps. During development, speed of iteration matters more than detection accuracy.

**Do not switch to yolov8l or yolov8x — they are too slow for CPU inference.**

---

## How to add a new decision

When a significant architectural choice is made during development, 
Claude should add it here in the same format:
- What was decided
- Why
- What should NOT be done as a result
