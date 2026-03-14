"""
Annotate the fixture video with YOLOv8 tracking + clothing classifications.

Each person is assigned a persistent tracker ID. Classification runs once per
new ID and is cached — labels stay stable even when detection wobbles at frame
edges.

Reads  : tests/fixtures/one.mov
Writes : tests/fixtures/annotated_one.mp4

Usage:
    python scripts/annotate_video.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.classifier import ClassificationResult, ClassifierConfig, ClothingClassifier
from pipeline.detector import DetectorConfig, PersonDetector, TrackedBox

INPUT_VIDEO = PROJECT_ROOT / "tests" / "fixtures" / "one.mov"
OUTPUT_VIDEO = PROJECT_ROOT / "tests" / "fixtures" / "annotated_one.mp4"

# Stable per-track-ID palette — up to 30 tracks, cycling after that.
_PALETTE = [
    (255, 56,  56),  (255, 157, 151), (255, 112, 31),  (255, 178, 29),
    (207, 210, 49),  (72,  249, 10),  (146, 204, 23),  (61,  219, 134),
    (26,  147, 52),  (0,   212, 187), (44,  153, 168),  (0,   194, 255),
    (52,  69,  147), (100, 115, 255), (0,   24,  236),  (132, 56,  255),
    (82,  0,   133), (203, 56,  255), (255, 149, 200),  (255, 55,  199),
    (255, 255, 0),   (0,   255, 255), (255, 0,   255),  (0,   165, 255),
    (128, 0,   128), (0,   128, 0),   (128, 128, 0),    (0,   0,   128),
    (128, 0,   0),   (0,   128, 128),
]


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Return a stable BGR colour for a given track ID."""
    r, g, b = _PALETTE[track_id % len(_PALETTE)]
    return (b, g, r)  # OpenCV is BGR


def _draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    bg_color: tuple[int, int, int],
) -> None:
    """Draw a filled-background text label at (x, y)."""
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.52
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad    = 3
    y_text = max(th + pad * 2, y)
    cv2.rectangle(frame, (x, y_text - th - pad * 2), (x + tw + pad * 2, y_text),
                  bg_color, cv2.FILLED)
    b, g, r    = bg_color
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)
    cv2.putText(frame, text, (x + pad, y_text - pad), font, scale,
                text_color, thickness, cv2.LINE_AA)


def annotate(detector: PersonDetector, classifier: ClothingClassifier) -> None:
    """Process every frame: track persons, classify once per track, draw labels."""
    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {INPUT_VIDEO}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, src_fps, (width, height))

    # Cache: track_id → ClassificationResult  (classify once per new ID)
    track_cache: dict[int, ClassificationResult] = {}

    frame_idx = 0
    print(f"Processing {total} frames at {src_fps:.1f} fps …")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracked: list[TrackedBox] = detector.track_array(frame)

        for box in tracked:
            tid = box.track_id

            # Classify the person the first time we see this track ID.
            if tid not in track_cache:
                crop = frame[box.y1:box.y2, box.x1:box.x2]
                if crop.size > 0:
                    try:
                        track_cache[tid] = classifier.classify(crop)
                    except Exception as exc:
                        logger_msg = f"classify failed for track {tid}: {exc}"
                        print(f"  WARNING: {logger_msg}")
                        continue

            result = track_cache.get(tid)
            if result is None:
                continue

            color = _track_color(tid)

            # Bounding box — thickness 2
            cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), color, 2)

            # Main label: "ID3 blue jacket"
            acc_str = ", ".join(a.value for a in result.accessories
                                if a.value != "none")
            label = f"#{tid} {result.dominant_color.value} {result.garment_type.value}"
            if acc_str:
                label += f" | {acc_str}"
            _draw_label(frame, label, box.x1, box.y1, color)

            # Confidence badge bottom-right of box
            _draw_label(frame, f"{box.confidence:.2f}", box.x2 - 46, box.y2, color)

        # Timestamp top-left
        ts = frame_idx / src_fps
        cv2.putText(frame, f"t={ts:.2f}s  frame={frame_idx}  tracks={len(tracked)}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{total} frames  |  {len(track_cache)} unique tracks so far …")

    cap.release()
    writer.release()
    print(f"\nUnique persons tracked: {len(track_cache)}")
    print(f"Saved → {OUTPUT_VIDEO}")


def main() -> None:
    print("Loading models …")
    detector   = PersonDetector(DetectorConfig(model_name="yolov8n.pt", confidence_threshold=0.4))
    classifier = ClothingClassifier(ClassifierConfig())
    print("Models ready.\n")
    annotate(detector, classifier)


if __name__ == "__main__":
    main()
