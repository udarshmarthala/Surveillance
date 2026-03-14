"""Extract keyframes from video files at a fixed rate using OpenCV."""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for frame extraction."""

    frame_storage_path: Path
    keyframe_fps: float = 2.0


@dataclass
class ExtractedFrame:
    """Metadata for a single saved frame."""

    frame_path: Path
    timestamp_seconds: float
    frame_index: int


def extract_frames(
    video_path: Path,
    camera_id: str,
    config: ExtractionConfig,
) -> list[ExtractedFrame]:
    """Open a video file and save one JPEG frame every 0.5 seconds."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    try:
        return _run_extraction(cap, video_path, camera_id, config)
    finally:
        cap.release()


def _run_extraction(
    cap: cv2.VideoCapture,
    video_path: Path,
    camera_id: str,
    config: ExtractionConfig,
) -> list[ExtractedFrame]:
    """Drive the frame-by-frame extraction loop and return saved frame metadata."""
    source_fps: float = cap.get(cv2.CAP_PROP_FPS)
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_fps <= 0:
        raise RuntimeError(f"Could not read FPS from video: {video_path}")

    # How many source frames to skip between each saved keyframe.
    frame_interval: int = max(1, round(source_fps / config.keyframe_fps))
    duration_s: float = total_frames / source_fps

    logger.info(
        "Starting extraction: video=%s camera=%s source_fps=%.2f "
        "duration=%.1fs keyframe_interval=%d",
        video_path.name,
        camera_id,
        source_fps,
        duration_s,
        frame_interval,
    )

    output_dir = _make_output_dir(config.frame_storage_path, camera_id, video_path)
    saved_frames: list[ExtractedFrame] = []
    source_frame_index: int = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if source_frame_index % frame_interval == 0:
            timestamp_s: float = source_frame_index / source_fps
            saved = _save_frame(frame, source_frame_index, timestamp_s, output_dir)
            if saved is not None:
                saved_frames.append(saved)

        source_frame_index += 1

    logger.info(
        "Extraction complete: video=%s saved=%d frames",
        video_path.name,
        len(saved_frames),
    )
    return saved_frames


def _make_output_dir(
    frame_storage_path: Path,
    camera_id: str,
    video_path: Path,
) -> Path:
    """Create and return the output directory for this video's frames."""
    output_dir = frame_storage_path / camera_id / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_frame(
    frame,
    source_frame_index: int,
    timestamp_s: float,
    output_dir: Path,
) -> ExtractedFrame | None:
    """Write a single frame as a JPEG and return its metadata, or None on failure."""
    filename = f"frame_{source_frame_index:06d}_{timestamp_s:.3f}s.jpg"
    frame_path = output_dir / filename

    try:
        success = cv2.imwrite(str(frame_path), frame)
        if not success:
            logger.error("cv2.imwrite returned False for %s", frame_path)
            return None
    except Exception as exc:
        logger.error("Failed to write frame %s: %s", frame_path, exc)
        return None

    logger.debug("Saved frame index=%d timestamp=%.3fs path=%s", source_frame_index, timestamp_s, frame_path)
    return ExtractedFrame(
        frame_path=frame_path,
        timestamp_seconds=timestamp_s,
        frame_index=source_frame_index,
    )
