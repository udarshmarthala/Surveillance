"""Detect persons in video frames using YOLOv8 and return bounding boxes."""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# YOLOv8 class index for "person" in the COCO dataset.
_PERSON_CLASS_ID = 0


@dataclass
class DetectorConfig:
    """Configuration for the person detector."""

    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.5


@dataclass
class BoundingBox:
    """Pixel-coordinate bounding box for a detected person."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@dataclass
class TrackedBox(BoundingBox):
    """Bounding box with a persistent tracker ID across frames."""

    track_id: int = -1


class PersonDetector:
    """Loads a YOLOv8 model once and runs person detection on JPEG frames."""

    def __init__(self, config: DetectorConfig) -> None:
        """Load the YOLOv8 model from disk. Raises if the model cannot be loaded."""
        self._config = config
        logger.info("Loading YOLO model: %s", config.model_name)
        try:
            self._model = YOLO(config.model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model '{config.model_name}': {exc}") from exc
        logger.info("YOLO model loaded successfully.")

    def detect(self, image_path: Path) -> list[BoundingBox]:
        """Run YOLOv8 on a JPEG file and return bounding boxes for persons only."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"OpenCV could not read image: {image_path}")

        return self._run_inference(image, image_path.name)

    def detect_array(self, image: np.ndarray) -> list[BoundingBox]:
        """Run YOLOv8 on an in-memory BGR numpy array and return person boxes."""
        return self._run_inference(image, "frame")

    def track_array(self, image: np.ndarray) -> list[TrackedBox]:
        """Run YOLOv8 + ByteTrack on a frame; returns boxes with persistent track IDs."""
        import torch

        with torch.no_grad():
            results = self._model.track(
                image,
                persist=True,
                conf=self._config.confidence_threshold,
                classes=[_PERSON_CLASS_ID],
                verbose=False,
                tracker="bytetrack.yaml",
            )

        boxes: list[TrackedBox] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                try:
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    track_id = int(box.id[0]) if box.id is not None else -1
                    boxes.append(
                        TrackedBox(x1=x1, y1=y1, x2=x2, y2=y2,
                                   confidence=confidence, track_id=track_id)
                    )
                except Exception as exc:
                    logger.error("Failed to parse tracked box: %s", exc)
        logger.debug("Tracked %d person(s) in frame", len(boxes))
        return boxes

    def _run_inference(self, image, image_name: str) -> list[BoundingBox]:
        """Run inference on an already-loaded image array and return person boxes."""
        import torch

        with torch.no_grad():
            results = self._model(
                image,
                conf=self._config.confidence_threshold,
                classes=[_PERSON_CLASS_ID],
                verbose=False,
            )

        boxes = _parse_results(results, image_name)
        logger.info("Detected %d person(s) in %s", len(boxes), image_name)
        return boxes


def _parse_results(results, image_name: str) -> list[BoundingBox]:
    """Extract BoundingBox objects from a YOLOv8 Results list."""
    boxes: list[BoundingBox] = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            try:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence))
                logger.debug(
                    "Person box: (%d,%d,%d,%d) conf=%.3f in %s",
                    x1, y1, x2, y2, confidence, image_name,
                )
            except Exception as exc:
                logger.error("Failed to parse box in %s: %s", image_name, exc)

    return boxes
