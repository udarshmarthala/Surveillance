"""
Demo: run extractor → detector → classifier on the fixture video.

Usage:
    python scripts/demo_pipeline.py
"""

import sys
import tempfile
import textwrap
from pathlib import Path

import cv2

# Make sure the project root is on the path when run from any directory.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.extractor import ExtractionConfig, extract_frames
from pipeline.classifier import ClassificationResult, ClassifierConfig, ClothingClassifier
from pipeline.detector import BoundingBox, DetectorConfig, PersonDetector

VIDEO = PROJECT_ROOT / "tests" / "fixtures" / "one.mov"
LINE = "─" * 60


def banner(text: str) -> None:
    print(f"\n{LINE}")
    print(f"  {text}")
    print(LINE)


def step1_extract(tmp_dir: Path) -> list:
    banner("STEP 1 — Frame Extraction (extractor.py)")
    config = ExtractionConfig(frame_storage_path=tmp_dir, keyframe_fps=2.0)
    frames = extract_frames(VIDEO, camera_id="cam_demo", config=config)
    print(f"  Video  : {VIDEO.name}")
    print(f"  Output : {tmp_dir / 'cam_demo' / VIDEO.stem}")
    print(f"  Frames saved : {len(frames)}")
    for f in frames[:5]:
        print(f"    [{f.frame_index:>6}]  t={f.timestamp_seconds:>6.2f}s  {f.frame_path.name}")
    if len(frames) > 5:
        print(f"    ... and {len(frames) - 5} more")
    return frames


def step2_detect(frames: list, detector: PersonDetector) -> list[tuple]:
    banner("STEP 2 — Person Detection (detector.py)")
    detections: list[tuple] = []   # (frame, box, crop_bgr)
    total_persons = 0

    for frame in frames:
        boxes = detector.detect(frame.frame_path)
        if not boxes:
            continue
        total_persons += len(boxes)
        image = cv2.imread(str(frame.frame_path))
        for box in boxes:
            crop = image[box.y1:box.y2, box.x1:box.x2]
            if crop.size > 0:
                detections.append((frame, box, crop))

    print(f"  Frames scanned : {len(frames)}")
    print(f"  Persons found  : {total_persons}")
    if detections:
        print("\n  Sample detections:")
        for frame, box, _ in detections[:5]:
            print(
                f"    t={frame.timestamp_seconds:.2f}s  "
                f"bbox=({box.x1},{box.y1})→({box.x2},{box.y2})  "
                f"conf={box.confidence:.2f}"
            )
        if len(detections) > 5:
            print(f"    ... and {len(detections) - 5} more")
    else:
        print("\n  No persons detected in this video clip.")
        print("  (This is fine — the fixture video may not contain people.)")

    return detections


def step3_classify(detections: list[tuple], classifier: ClothingClassifier) -> None:
    banner("STEP 3 — Clothing Classification (classifier.py)")

    if not detections:
        print("  No crops to classify — skipping.")
        return

    print(f"  Classifying {len(detections)} person crop(s)...\n")
    for i, (frame, box, crop_bgr) in enumerate(detections, start=1):
        result: ClassificationResult = classifier.classify(crop_bgr)
        accessories_str = ", ".join(a.value for a in result.accessories)
        print(
            f"  [{i:>2}] t={frame.timestamp_seconds:.2f}s  "
            f"color={result.dominant_color.value:<8}  "
            f"garment={result.garment_type.value:<8}  "
            f"accessories={accessories_str}"
        )


def main() -> None:
    print("\n" + "═" * 60)
    print("  SURVEILLANCE SEARCH — Pipeline Demo")
    print("═" * 60)

    # Load models once
    banner("Loading models…")
    detector = PersonDetector(DetectorConfig(model_name="yolov8n.pt", confidence_threshold=0.5))
    classifier = ClothingClassifier(ClassifierConfig())
    print("  ✓ YOLOv8n loaded")
    print("  ✓ CLIP (openai/clip-vit-base-patch32) loaded")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        frames = step1_extract(tmp_path)
        detections = step2_detect(frames, detector)
        step3_classify(detections, classifier)

    banner("Done")
    print("  All three pipeline stages completed successfully.\n")


if __name__ == "__main__":
    main()
