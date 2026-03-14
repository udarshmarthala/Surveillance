"""Tests for ingestion/extractor.py."""

import math
from pathlib import Path

import cv2
import pytest

from ingestion.extractor import (
    ExtractionConfig,
    ExtractedFrame,
    extract_frames,
)

FIXTURE_VIDEO = Path(__file__).parent / "fixtures" / "one.mov"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, keyframe_fps: float = 2.0) -> ExtractionConfig:
    """Return an ExtractionConfig pointing at pytest's tmp_path."""
    return ExtractionConfig(frame_storage_path=tmp_path, keyframe_fps=keyframe_fps)


def _video_duration(video_path: Path) -> float:
    """Return the duration of a video in seconds."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return total / fps


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractFrames:
    def test_returns_list_of_extracted_frames(self, tmp_path: Path) -> None:
        """extract_frames returns a non-empty list of ExtractedFrame objects."""
        config = _make_config(tmp_path)
        results = extract_frames(FIXTURE_VIDEO, "cam_01", config)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(f, ExtractedFrame) for f in results)

    def test_frame_count_matches_expected_fps(self, tmp_path: Path) -> None:
        """Number of saved frames is approximately duration * keyframe_fps."""
        keyframe_fps = 2.0
        config = _make_config(tmp_path, keyframe_fps=keyframe_fps)
        results = extract_frames(FIXTURE_VIDEO, "cam_01", config)

        duration_s = _video_duration(FIXTURE_VIDEO)
        expected = math.floor(duration_s * keyframe_fps) + 1  # +1 for frame 0
        # Allow ±2 frames of tolerance due to rounding in frame intervals.
        assert abs(len(results) - expected) <= 2

    def test_all_jpeg_files_exist_on_disk(self, tmp_path: Path) -> None:
        """Every ExtractedFrame path points to a real JPEG file on disk."""
        config = _make_config(tmp_path)
        results = extract_frames(FIXTURE_VIDEO, "cam_01", config)

        for frame in results:
            assert frame.frame_path.exists(), f"Missing file: {frame.frame_path}"
            assert frame.frame_path.suffix == ".jpg"

    def test_output_directory_structure(self, tmp_path: Path) -> None:
        """Frames are saved under frame_storage_path/camera_id/video_stem/."""
        config = _make_config(tmp_path)
        results = extract_frames(FIXTURE_VIDEO, "cam_01", config)

        expected_dir = tmp_path / "cam_01" / FIXTURE_VIDEO.stem
        assert expected_dir.is_dir()
        for frame in results:
            assert frame.frame_path.parent == expected_dir

    def test_timestamps_are_monotonically_increasing(self, tmp_path: Path) -> None:
        """Timestamps increase with each successive frame."""
        config = _make_config(tmp_path)
        results = extract_frames(FIXTURE_VIDEO, "cam_01", config)

        timestamps = [f.timestamp_seconds for f in results]
        assert timestamps == sorted(timestamps)
        # First frame is at time 0.
        assert timestamps[0] == pytest.approx(0.0, abs=0.05)

    def test_timestamps_are_spaced_roughly_half_second_apart(self, tmp_path: Path) -> None:
        """Gap between consecutive timestamps is approximately 0.5 seconds."""
        config = _make_config(tmp_path, keyframe_fps=2.0)
        results = extract_frames(FIXTURE_VIDEO, "cam_01", config)

        gaps = [
            results[i + 1].timestamp_seconds - results[i].timestamp_seconds
            for i in range(len(results) - 1)
        ]
        for gap in gaps:
            assert gap == pytest.approx(0.5, abs=0.1), f"Unexpected gap: {gap}"

    def test_jpegs_are_valid_images(self, tmp_path: Path) -> None:
        """Each saved JPEG can be re-read by OpenCV and has non-zero dimensions."""
        config = _make_config(tmp_path)
        results = extract_frames(FIXTURE_VIDEO, "cam_01", config)

        for frame in results:
            img = cv2.imread(str(frame.frame_path))
            assert img is not None, f"Could not read {frame.frame_path}"
            h, w = img.shape[:2]
            assert h > 0 and w > 0

    def test_different_camera_ids_produce_separate_directories(
        self, tmp_path: Path
    ) -> None:
        """Two calls with different camera_ids write to separate subdirectories."""
        config = _make_config(tmp_path)
        extract_frames(FIXTURE_VIDEO, "cam_01", config)
        extract_frames(FIXTURE_VIDEO, "cam_02", config)

        assert (tmp_path / "cam_01").is_dir()
        assert (tmp_path / "cam_02").is_dir()

    def test_missing_video_raises_file_not_found(self, tmp_path: Path) -> None:
        """extract_frames raises FileNotFoundError for a non-existent video."""
        config = _make_config(tmp_path)
        with pytest.raises(FileNotFoundError):
            extract_frames(Path("/nonexistent/video.mp4"), "cam_01", config)

    def test_custom_keyframe_fps(self, tmp_path: Path) -> None:
        """Setting keyframe_fps=1 produces roughly half as many frames as fps=2."""
        config_2fps = _make_config(tmp_path / "2fps", keyframe_fps=2.0)
        config_1fps = _make_config(tmp_path / "1fps", keyframe_fps=1.0)

        results_2fps = extract_frames(FIXTURE_VIDEO, "cam_01", config_2fps)
        results_1fps = extract_frames(FIXTURE_VIDEO, "cam_01", config_1fps)

        # 1fps should yield roughly half the frames of 2fps (±2 tolerance).
        assert abs(len(results_1fps) - len(results_2fps) // 2) <= 2
