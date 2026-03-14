"""Tests for pipeline/detector.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.detector import BoundingBox, DetectorConfig, PersonDetector

FIXTURE_FRAME = Path(__file__).parent / "fixtures" / "sample_frame.jpg"


# ---------------------------------------------------------------------------
# Helpers — build a fake YOLO result to avoid running real inference
# ---------------------------------------------------------------------------

def _fake_box(x1: int, y1: int, x2: int, y2: int, conf: float, cls: int = 0):
    """Return a mock object shaped like a single ultralytics box."""
    import torch

    box = MagicMock()
    box.xyxy = [torch.tensor([x1, y1, x2, y2], dtype=torch.float32)]
    box.conf = [torch.tensor(conf)]
    box.cls = [torch.tensor(cls)]
    return box


def _fake_results(boxes_data: list) -> list:
    """Return a mock Results list with the given box mocks."""
    result = MagicMock()
    result.boxes = boxes_data if boxes_data is not None else None
    return [result]


# ---------------------------------------------------------------------------
# Tests — structural / contract
# ---------------------------------------------------------------------------

class TestPersonDetectorContract:
    def test_returns_list_of_bounding_boxes(self) -> None:
        """detect() always returns a list (may be empty if no persons found)."""
        config = DetectorConfig(confidence_threshold=0.5)
        detector = PersonDetector(config)
        result = detector.detect(FIXTURE_FRAME)

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, BoundingBox)

    def test_bounding_box_fields_are_populated(self) -> None:
        """Every BoundingBox has non-negative int coordinates and a float confidence."""
        config = DetectorConfig(confidence_threshold=0.3)
        detector = PersonDetector(config)
        results = detector.detect(FIXTURE_FRAME)

        for box in results:
            assert isinstance(box.x1, int)
            assert isinstance(box.y1, int)
            assert isinstance(box.x2, int)
            assert isinstance(box.y2, int)
            assert isinstance(box.confidence, float)
            assert box.x1 >= 0 and box.y1 >= 0
            assert box.x2 > box.x1
            assert box.y2 > box.y1

    def test_confidence_values_within_range(self) -> None:
        """All returned confidence values are between 0 and 1."""
        config = DetectorConfig(confidence_threshold=0.3)
        detector = PersonDetector(config)
        results = detector.detect(FIXTURE_FRAME)

        for box in results:
            assert 0.0 <= box.confidence <= 1.0

    def test_missing_image_raises_file_not_found(self) -> None:
        """detect() raises FileNotFoundError for a missing image path."""
        detector = PersonDetector(DetectorConfig())
        with pytest.raises(FileNotFoundError):
            detector.detect(Path("/nonexistent/frame.jpg"))


# ---------------------------------------------------------------------------
# Tests — confidence filtering (mocked inference)
# ---------------------------------------------------------------------------

class TestConfidenceFiltering:
    """Verify that confidence thresholds are respected, using mocked YOLO output."""

    def _detector_with_fake_results(self, fake_results, confidence_threshold: float = 0.5):
        """Return a PersonDetector whose _model() returns fake_results."""
        config = DetectorConfig(confidence_threshold=confidence_threshold)
        detector = PersonDetector(config)
        detector._model = MagicMock(return_value=fake_results)
        return detector

    def test_boxes_above_threshold_are_returned(self) -> None:
        """Boxes with confidence above the threshold appear in results."""
        fake = _fake_results([_fake_box(10, 20, 100, 200, conf=0.9)])
        detector = self._detector_with_fake_results(fake, confidence_threshold=0.5)

        results = detector.detect(FIXTURE_FRAME)
        assert len(results) == 1
        assert results[0].confidence == pytest.approx(0.9, abs=0.001)

    def test_empty_result_when_no_detections(self) -> None:
        """An empty boxes list results in an empty return list."""
        fake = _fake_results([])
        detector = self._detector_with_fake_results(fake)

        results = detector.detect(FIXTURE_FRAME)
        assert results == []

    def test_none_boxes_attribute_handled_gracefully(self) -> None:
        """A result whose .boxes is None does not crash the parser."""
        fake = _fake_results(None)
        detector = self._detector_with_fake_results(fake)

        results = detector.detect(FIXTURE_FRAME)
        assert results == []

    def test_multiple_persons_all_returned(self) -> None:
        """Multiple detected persons all appear in the result list."""
        fake = _fake_results([
            _fake_box(0, 0, 50, 100, conf=0.8),
            _fake_box(200, 100, 300, 400, conf=0.7),
            _fake_box(400, 0, 500, 200, conf=0.6),
        ])
        detector = self._detector_with_fake_results(fake)

        results = detector.detect(FIXTURE_FRAME)
        assert len(results) == 3

    def test_bounding_box_coordinates_are_integers(self) -> None:
        """Coordinates in returned BoundingBox objects are Python ints."""
        fake = _fake_results([_fake_box(10, 20, 110, 220, conf=0.75)])
        detector = self._detector_with_fake_results(fake)

        results = detector.detect(FIXTURE_FRAME)
        box = results[0]
        assert box.x1 == 10
        assert box.y1 == 20
        assert box.x2 == 110
        assert box.y2 == 220


# ---------------------------------------------------------------------------
# Tests — model loading
# ---------------------------------------------------------------------------

class TestModelLoading:
    def test_bad_model_name_raises_runtime_error(self) -> None:
        """A model name that YOLO cannot resolve raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Failed to load YOLO model"):
            PersonDetector(DetectorConfig(model_name="nonexistent_model_xyz.pt"))
