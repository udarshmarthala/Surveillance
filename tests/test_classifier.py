"""Tests for pipeline/classifier.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline.classifier import (
    Accessory,
    ClassificationResult,
    ClassifierConfig,
    ClothingClassifier,
    ClothingColor,
    GarmentType,
    _hls_to_color,
    classify_color,
)


# ---------------------------------------------------------------------------
# Helpers — build synthetic BGR images with known HSL properties
# ---------------------------------------------------------------------------

def _solid_bgr(b: int, g: int, r: int, size: int = 50) -> np.ndarray:
    """Return a solid-color BGR image of shape (size, size, 3)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (b, g, r)
    return img


def _hls_to_bgr_pixel(h_opencv: float, l_val: float, s_val: float) -> tuple[int, int, int]:
    """Convert a single HLS pixel (OpenCV scale) to BGR for building test images."""
    import cv2
    hls = np.array([[[int(h_opencv), int(l_val), int(s_val)]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
    b, g, r = int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])
    return b, g, r


# ---------------------------------------------------------------------------
# Tests — _hls_to_color (unit tests for each hue range from CLAUDE.md)
# ---------------------------------------------------------------------------

class TestHlsToColor:
    """Directly test the hue-range mapping function with precise HLS values."""

    # Achromatic cases
    def test_black_low_lightness(self) -> None:
        assert _hls_to_color(0, 30, 200) == ClothingColor.BLACK

    def test_white_high_lightness(self) -> None:
        assert _hls_to_color(0, 210, 10) == ClothingColor.WHITE

    def test_grey_low_saturation(self) -> None:
        assert _hls_to_color(90, 128, 20) == ClothingColor.GREY

    # Chromatic hue ranges (hue_degrees = h_opencv * 2)
    # Red: 0–15° and 345–360°  → h_opencv 0–7.5 and 172.5–180
    def test_red_low_end(self) -> None:
        assert _hls_to_color(3, 128, 200) == ClothingColor.RED   # 6°

    def test_red_high_end(self) -> None:
        assert _hls_to_color(175, 128, 200) == ClothingColor.RED  # 350°

    # Orange: 15–45° → h_opencv 7.5–22.5
    def test_orange(self) -> None:
        assert _hls_to_color(15, 128, 200) == ClothingColor.ORANGE  # 30°

    # Yellow: 45–75° → h_opencv 22.5–37.5
    def test_yellow(self) -> None:
        assert _hls_to_color(30, 128, 200) == ClothingColor.YELLOW  # 60°

    # Green: 75–150° → h_opencv 37.5–75
    def test_green(self) -> None:
        assert _hls_to_color(55, 128, 200) == ClothingColor.GREEN   # 110°

    # Blue: 150–255° → h_opencv 75–127.5
    def test_blue(self) -> None:
        assert _hls_to_color(110, 128, 200) == ClothingColor.BLUE   # 220°

    # Purple: 255–315° → h_opencv 127.5–157.5
    def test_purple(self) -> None:
        assert _hls_to_color(145, 128, 200) == ClothingColor.PURPLE  # 290°

    # Pink: 315–345° → h_opencv 157.5–172.5
    def test_pink(self) -> None:
        assert _hls_to_color(165, 128, 200) == ClothingColor.PINK   # 330°

    # Boundary: exactly at 15° (orange side)
    def test_boundary_red_orange(self) -> None:
        # h_opencv=7.5 → 15°, which is the start of orange
        result = _hls_to_color(7.5, 128, 200)
        assert result == ClothingColor.ORANGE


# ---------------------------------------------------------------------------
# Tests — classify_color (integration: real image → HLS → color)
# ---------------------------------------------------------------------------

class TestClassifyColor:
    """Test classify_color() with synthetic solid-color BGR images."""

    def test_black_image(self) -> None:
        img = _solid_bgr(10, 10, 10)
        assert classify_color(img) == ClothingColor.BLACK

    def test_white_image(self) -> None:
        img = _solid_bgr(240, 240, 240)
        assert classify_color(img) == ClothingColor.WHITE

    def test_grey_image(self) -> None:
        img = _solid_bgr(128, 128, 128)
        assert classify_color(img) == ClothingColor.GREY

    def test_vivid_red(self) -> None:
        img = _solid_bgr(0, 0, 255)   # pure red in BGR
        assert classify_color(img) == ClothingColor.RED

    def test_vivid_blue(self) -> None:
        img = _solid_bgr(255, 0, 0)   # pure blue in BGR
        assert classify_color(img) == ClothingColor.BLUE

    def test_vivid_green(self) -> None:
        img = _solid_bgr(0, 200, 0)   # green in BGR
        assert classify_color(img) == ClothingColor.GREEN

    def test_vivid_yellow(self) -> None:
        img = _solid_bgr(0, 255, 255) # yellow in BGR
        assert classify_color(img) == ClothingColor.YELLOW

    def test_returns_clothing_color_enum(self) -> None:
        img = _solid_bgr(128, 0, 128)
        result = classify_color(img)
        assert isinstance(result, ClothingColor)

    def test_empty_image_raises(self) -> None:
        with pytest.raises((cv2.error, Exception)):
            classify_color(np.zeros((0, 0, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Tests — ClothingClassifier (mocked CLIP to avoid downloading the model)
# ---------------------------------------------------------------------------

def _make_mock_classifier(
    garment_idx: int = 0,
    detected_accessories: list | None = None,
) -> ClothingClassifier:
    """
    Return a ClothingClassifier whose CLIP model is replaced with a mock.
    garment_idx: which GarmentType index the mock will pick.
    detected_accessories: list of Accessory values the mock will "detect".
      Each accessory gets a binary yes/no CLIP call; those in this list return
      positive score >= threshold, the rest return negative score.
    """
    detected_accessories = detected_accessories or []

    with patch("pipeline.classifier.CLIPModel.from_pretrained"), \
         patch("pipeline.classifier.CLIPProcessor.from_pretrained"):
        classifier = ClothingClassifier(ClassifierConfig())

    from pipeline.classifier import GarmentType as GT, _ACCESSORY_BINARY_CHECKS

    n_garments = len(list(GT))

    def fake_clip_scores(image, prompts):
        n = len(prompts)
        scores = np.zeros(n, dtype=np.float32)
        if n == n_garments:
            scores[garment_idx] = 1.0
        elif n == 2:
            # Binary accessory check: match by positive prompt string.
            for acc, pos_prompt, _ in _ACCESSORY_BINARY_CHECKS:
                if prompts[0] == pos_prompt:
                    if acc in detected_accessories:
                        scores[0] = 0.9   # positive wins → detected
                    else:
                        scores[1] = 0.9   # negative wins → not detected
                    break
        return scores

    classifier._clip_scores = fake_clip_scores
    return classifier


class TestClothingClassifier:
    def test_classify_returns_classification_result(self) -> None:
        """classify() returns a ClassificationResult instance."""
        classifier = _make_mock_classifier()
        img = _solid_bgr(0, 0, 255)  # red
        result = classifier.classify(img)
        assert isinstance(result, ClassificationResult)

    def test_classify_dominant_color_is_clothing_color_enum(self) -> None:
        classifier = _make_mock_classifier()
        result = classifier.classify(_solid_bgr(0, 0, 255))
        assert isinstance(result.dominant_color, ClothingColor)

    def test_classify_garment_type_is_enum(self) -> None:
        classifier = _make_mock_classifier(garment_idx=1)  # jacket
        result = classifier.classify(_solid_bgr(0, 0, 255))
        assert isinstance(result.garment_type, GarmentType)

    def test_classify_garment_picks_correct_index(self) -> None:
        garments = list(GarmentType)
        for idx, expected in enumerate(garments):
            classifier = _make_mock_classifier(garment_idx=idx)
            result = classifier.classify(_solid_bgr(128, 128, 128))
            assert result.garment_type == expected

    def test_classify_accessories_is_list(self) -> None:
        classifier = _make_mock_classifier()
        result = classifier.classify(_solid_bgr(128, 128, 128))
        assert isinstance(result.accessories, list)

    def test_no_accessories_returns_none_label(self) -> None:
        classifier = _make_mock_classifier(detected_accessories=[])
        result = classifier.classify(_solid_bgr(128, 128, 128))
        assert result.accessories == [Accessory.NONE]

    def test_glasses_detected(self) -> None:
        classifier = _make_mock_classifier(detected_accessories=[Accessory.GLASSES])
        result = classifier.classify(_solid_bgr(128, 128, 128))
        assert Accessory.GLASSES in result.accessories

    def test_hat_detected(self) -> None:
        classifier = _make_mock_classifier(detected_accessories=[Accessory.HAT])
        result = classifier.classify(_solid_bgr(128, 128, 128))
        assert Accessory.HAT in result.accessories

    def test_bag_detected(self) -> None:
        classifier = _make_mock_classifier(detected_accessories=[Accessory.BAG])
        result = classifier.classify(_solid_bgr(128, 128, 128))
        assert Accessory.BAG in result.accessories

    def test_multiple_accessories_detected_simultaneously(self) -> None:
        """Binary checks allow detecting glasses AND hat on the same person."""
        classifier = _make_mock_classifier(
            detected_accessories=[Accessory.GLASSES, Accessory.HAT]
        )
        result = classifier.classify(_solid_bgr(128, 128, 128))
        assert Accessory.GLASSES in result.accessories
        assert Accessory.HAT in result.accessories
        assert Accessory.NONE not in result.accessories

    def test_empty_image_raises_value_error(self) -> None:
        classifier = _make_mock_classifier()
        with pytest.raises(ValueError, match="empty image"):
            classifier.classify(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_color_is_correct_for_solid_blue_crop(self) -> None:
        """End-to-end: blue crop → dominant_color == BLUE."""
        classifier = _make_mock_classifier()
        result = classifier.classify(_solid_bgr(255, 0, 0))  # BGR pure blue
        assert result.dominant_color == ClothingColor.BLUE

    def test_bad_model_name_raises_runtime_error(self) -> None:
        with pytest.raises(RuntimeError, match="Failed to load CLIP model"):
            ClothingClassifier(ClassifierConfig(clip_model_name="nonexistent/model"))


# Keep cv2 importable inside TestClassifyColor
import cv2  # noqa: E402  (needed for the exception type in test_empty_image_raises)
