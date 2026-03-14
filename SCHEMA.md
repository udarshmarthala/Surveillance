# SCHEMA.md — Data Schema Reference

Claude must read this file before implementing any file in `storage/`, `pipeline/`, or `search/`.
This is the single source of truth for column names, types, and enum values.
All implementation files must conform to what is defined here.

If a schema decision needs to change, update this file first, log the change in DECISIONS.md, then update the code.

---

## SQLite table: `detections`

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `id` | INTEGER | No | Primary key, autoincrement |
| `frame_id` | TEXT | No | Filename of the source JPEG frame (e.g. `cam_01_000042.jpg`) |
| `camera_id` | TEXT | No | Caller-supplied identifier string (e.g. `cam_01`) |
| `timestamp_sec` | REAL | No | Seconds from the start of the source video |
| `bbox_x1` | REAL | No | Bounding box top-left x, pixel coordinates |
| `bbox_y1` | REAL | No | Bounding box top-left y, pixel coordinates |
| `bbox_x2` | REAL | No | Bounding box bottom-right x, pixel coordinates |
| `bbox_y2` | REAL | No | Bounding box bottom-right y, pixel coordinates |
| `color` | TEXT | No | Dominant clothing color — must be one of the Color enum values below |
| `garment` | TEXT | No | Garment type — must be one of the Garment enum values below |
| `accessories` | TEXT | No | Comma-separated accessory values, or the literal string `"none"` |
| `thumbnail_path` | TEXT | No | Relative path to the cropped person JPEG under `FRAME_STORAGE_PATH` |
| `confidence` | REAL | No | YOLOv8 detection confidence score (0.0–1.0) |
| `created_at` | TEXT | No | ISO 8601 datetime string (e.g. `2026-03-13T14:22:00`) |

**Index:** Add an index on `(camera_id, color, garment)` for fast SQL filtering in the search layer.

---

## ChromaDB collection: `person_crops`

| Field | Type | Notes |
|---|---|---|
| `id` | string | Matches `detections.id` cast to string (e.g. `"42"`) |
| `embedding` | list[float], len=512 | CLIP image encoder output for the person crop |
| `metadata.camera_id` | string | Same as `detections.camera_id` |
| `metadata.timestamp_sec` | float | Same as `detections.timestamp_sec` |
| `metadata.color` | string | Same as `detections.color` |
| `metadata.garment` | string | Same as `detections.garment` |
| `metadata.accessories` | string | Same as `detections.accessories` (comma-separated string) |

---

## Enum values — canonical strings

These are the only allowed values. Use these exact strings in the database, query parser output, and classifier output. Do not use synonyms or abbreviations.

### Color
```
red | orange | yellow | green | blue | purple | pink | black | white | grey | brown
```

HSL hue mapping (from CLAUDE.md color detection rules):
- `red`: 0–15° and 345–360°
- `orange`: 15–45°
- `yellow`: 45–75°
- `green`: 75–150°
- `blue`: 150–255°
- `purple`: 255–315°
- `pink`: 315–345°
- `black` / `white` / `grey` / `brown`: classified by lightness and saturation, not hue

### Garment
```
shirt | jacket | dress | shorts | pants
```

### Accessories
```
glasses | hat | bag | none
```

Stored in DB as comma-separated string. Examples:
- Single: `"glasses"`
- Multiple: `"glasses,hat"`
- None detected: `"none"`

---

## Search result object

This is the shape of each item in the `results` array returned by `POST /search` (see `API_CONTRACT.md`).

```json
{
  "detection_id": 42,
  "camera_id": "cam_01",
  "timestamp_sec": 137.5,
  "thumbnail_path": "cam_01/cam_01_000042_person_0.jpg",
  "color": "red",
  "garment": "shirt",
  "accessories": ["glasses"],
  "score": 0.83
}
```

- `score` = `0.6 * vector_score + 0.4 * attribute_score` — higher is better match
- `accessories` is always a `list[str]` in the API response, even if stored as a comma-separated string in the DB
- `thumbnail_path` is a relative path — prefix with `/thumbnails/` to form the full API URL

---

## Query parser output shape

`query_parser.py` must return a dict with exactly these keys:

```python
{
    "color": str | None,           # one of the Color enum values, or None if not found in query
    "garment": str | None,         # one of the Garment enum values, or None if not found
    "accessories": list[str],      # subset of Accessories enum values, empty list if none found
    "embedding": list[float],      # 512-dim CLIP text embedding of the original query string
}
```

`searcher.py` consumes exactly this dict. Do not add or rename keys without updating both files.
