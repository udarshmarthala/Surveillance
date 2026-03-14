# API_CONTRACT.md — FastAPI Endpoint Contract

Claude must read this file before implementing `search/api.py` or any frontend component.
This is the single source of truth for endpoint URLs, request shapes, and response shapes.

If an endpoint needs to change, update this file first, then update `api.py` and any affected frontend components together.

---

## Base URL

```
http://localhost:8000
```

All endpoints are relative to this base during development.

---

## Endpoints

### `POST /search`

Find persons matching a clothing description.

**Request body (JSON):**
```json
{
  "query": "red shirt with glasses",
  "top_k": 20,
  "camera_id": "cam_01"
}
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `query` | string | Yes | — | Natural language clothing description |
| `top_k` | integer | No | 20 | Max number of results to return |
| `camera_id` | string | No | null | If provided, restrict results to this camera only |

**Response body (JSON):**
```json
{
  "results": [
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
  ],
  "query_parsed": {
    "color": "red",
    "garment": "shirt",
    "accessories": ["glasses"]
  },
  "total": 1
}
```

- `results` is ordered by `score` descending (best match first)
- `thumbnail_path` is relative — use with `GET /thumbnails/{path}` to fetch the image
- `query_parsed` reflects what the parser extracted; useful for UI debugging (show user "searching for: red shirt + glasses")
- `accessories` in each result is always a `list[str]`, never a comma-separated string

**Errors:**
| Status | Condition |
|---|---|
| 422 | `query` field is missing or empty string |
| 500 | Internal search failure — response body: `{ "detail": "<error message>" }` |

---

### `GET /thumbnails/{path:path}`

Serve a person crop thumbnail image.

**URL example:**
```
GET /thumbnails/cam_01/cam_01_000042_person_0.jpg
```

**Response:**
- `200 OK` — `Content-Type: image/jpeg`, body is the JPEG file
- `404 Not Found` — if the file does not exist under `FRAME_STORAGE_PATH`

The `path` parameter is the value of `thumbnail_path` from a search result. Prefix it with `/thumbnails/` to form the full URL.

---

### `GET /health`

Check that the API server is running.

**Response:**
```json
{ "status": "ok" }
```

Always returns `200 OK`. No authentication. Used to verify the server is up before running tests or the frontend.

---

## What does NOT exist yet

Do not implement these until the three endpoints above are complete and tested:

- Video streaming endpoint
- Camera listing endpoint
- Ingest/upload endpoint (pipeline is triggered by scripts, not HTTP)
- Pagination (use `top_k` for now)
- Authentication

---

## Frontend usage reference

### SearchBar — POST /search
```js
const response = await fetch('/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query, top_k: 20 })
})
const data = await response.json()
// data.results — array of result objects
// data.query_parsed — what was extracted from the query
```

### ResultGrid — render each result
```jsx
// thumbnail_path from result → full image URL:
const imgUrl = `/thumbnails/${result.thumbnail_path}`

// display fields:
result.detection_id    // integer
result.camera_id       // string
result.timestamp_sec   // float — format as MM:SS for display
result.color           // string
result.garment         // string
result.accessories     // string[] — join with ", " for display
result.score           // float 0–1 — show as percentage
```

### Health check
```js
const res = await fetch('/health')
// res.ok === true means server is running
```
