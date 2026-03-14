"""Classify clothing attributes (color, garment type, accessories) from person crops using CLIP."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# String enums — these map to the values stored in the DB (SCHEMA.md)
# ---------------------------------------------------------------------------

class ClothingColor(str, Enum):
    RED = "red"
    ORANGE = "orange"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    PURPLE = "purple"
    PINK = "pink"
    BLACK = "black"
    WHITE = "white"
    GREY = "grey"
    BROWN = "brown"


class GarmentType(str, Enum):
    SHIRT = "shirt"
    JACKET = "jacket"
    DRESS = "dress"
    SHORTS = "shorts"
    PANTS = "pants"


class Accessory(str, Enum):
    GLASSES = "glasses"
    HAT = "hat"
    BAG = "bag"
    NONE = "none"


# ---------------------------------------------------------------------------
# Hue-to-color mapping (HSL hue in degrees, 0–360)
# Ranges from CLAUDE.md:
#   0–15 and 345–360 = red
#   15–45  = orange
#   45–75  = yellow
#   75–150 = green
#   150–255 = blue
#   255–315 = purple
#   315–345 = pink
# Achromatic detection (black/white/grey) is done via lightness + saturation.
# ---------------------------------------------------------------------------

# Each entry: (hue_min_inclusive, hue_max_exclusive, ClothingColor)
_HUE_RANGES: list[tuple[float, float, ClothingColor]] = [
    (0.0, 15.0, ClothingColor.RED),
    (15.0, 45.0, ClothingColor.ORANGE),
    (45.0, 75.0, ClothingColor.YELLOW),
    (75.0, 150.0, ClothingColor.GREEN),
    (150.0, 255.0, ClothingColor.BLUE),
    (255.0, 315.0, ClothingColor.PURPLE),
    (315.0, 345.0, ClothingColor.PINK),
    (345.0, 360.0, ClothingColor.RED),  # red wraps around
]

# Lightness and saturation thresholds (OpenCV HLS: H 0–180, L 0–255, S 0–255)
_BLACK_LIGHTNESS_MAX = 50   # L < 50/255 → black
_WHITE_LIGHTNESS_MIN = 200  # L > 200/255 → white
_GREY_SATURATION_MAX = 40   # S < 40/255 → grey (after ruling out black/white)

# CLIP zero-shot prompt templates
# Upper-body specific prompts reduce confusion from legs/pants below the torso.
_GARMENT_PROMPTS = [
    "a person wearing a shirt or t-shirt",
    "a person wearing a jacket or coat",
    "a person wearing a dress",
    "a person wearing shorts",
    "a person wearing trousers or pants",
]

# Binary per-accessory checks: avoids softmax competition where "bag" beats "none".
# Each entry: (Accessory, positive_prompt, negative_prompt)
_ACCESSORY_BINARY_CHECKS: list[tuple[Accessory, str, str]] = [
    (Accessory.GLASSES, "a person wearing glasses or sunglasses", "a person not wearing glasses"),
    (Accessory.HAT,     "a person wearing a hat or cap",          "a person not wearing a hat"),
    (Accessory.BAG,     "a person carrying a bag or backpack",    "a person not carrying a bag"),
]
# Positive score must exceed this to report the accessory as detected.
_ACCESSORY_BINARY_THRESHOLD = 0.60

# Fraction of crop height used for torso (colour) and upper-body (garment) regions.
# Skipping top 15% avoids face/hair; stopping at 75% avoids legs/feet.
_TORSO_Y_START = 0.15
_TORSO_Y_END   = 0.75
_TORSO_X_START = 0.10   # trim side background bleed
_TORSO_X_END   = 0.90
# CLIP garment: use only the upper half of the crop to avoid pants biasing shirt/jacket.
_UPPER_Y_END   = 0.55


# ---------------------------------------------------------------------------
# Config and result types
# ---------------------------------------------------------------------------

@dataclass
class ClassifierConfig:
    """Configuration for the clothing classifier."""

    clip_model_name: str = "openai/clip-vit-base-patch32"


@dataclass
class ClassificationResult:
    """Clothing attributes extracted from a single person crop."""

    dominant_color: ClothingColor
    garment_type: GarmentType
    accessories: list[Accessory] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ClothingClassifier:
    """Loads CLIP once and classifies color, garment, and accessories from crops."""

    def __init__(self, config: ClassifierConfig) -> None:
        """Load CLIP model and processor. Raises if loading fails."""
        logger.info("Loading CLIP model: %s", config.clip_model_name)
        try:
            self._processor = CLIPProcessor.from_pretrained(
                config.clip_model_name, local_files_only=True
            )
            self._model = CLIPModel.from_pretrained(
                config.clip_model_name, local_files_only=True
            )
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load CLIP model '{config.clip_model_name}': {exc}"
            ) from exc
        logger.info("CLIP model loaded successfully.")

    def classify(self, crop_bgr: np.ndarray) -> ClassificationResult:
        """Return dominant color, garment type, and accessories for a person crop."""
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Received an empty image array.")

        torso = _torso_region(crop_bgr)
        upper = _upper_body_region(crop_bgr)

        # Colour: torso region only (avoids face/hair/background)
        dominant_color = classify_color(torso)
        # Garment: upper-body crop → less confusion from pants below the waist
        garment_type = self._classify_garment(_bgr_to_pil(upper))
        # Accessories: full crop keeps hat (top) and bag (side/bottom) in frame
        accessories = self._classify_accessories(_bgr_to_pil(crop_bgr))

        logger.debug(
            "Classification: color=%s garment=%s accessories=%s",
            dominant_color.value,
            garment_type.value,
            [a.value for a in accessories],
        )
        return ClassificationResult(
            dominant_color=dominant_color,
            garment_type=garment_type,
            accessories=accessories,
        )

    def _classify_garment(self, image: Image.Image) -> GarmentType:
        """Use CLIP zero-shot to pick the most likely garment type."""
        scores = self._clip_scores(image, _GARMENT_PROMPTS)
        best_idx = int(scores.argmax())
        return list(GarmentType)[best_idx]

    def _classify_accessories(self, image: Image.Image) -> list[Accessory]:
        """Binary CLIP check per accessory; multiple can be detected simultaneously."""
        found: list[Accessory] = []
        for accessory, pos_prompt, neg_prompt in _ACCESSORY_BINARY_CHECKS:
            scores = self._clip_scores(image, [pos_prompt, neg_prompt])
            if float(scores[0]) >= _ACCESSORY_BINARY_THRESHOLD:
                found.append(accessory)
        return found if found else [Accessory.NONE]

    def _clip_scores(self, image: Image.Image, prompts: list[str]) -> np.ndarray:
        """Run CLIP on one image against a list of text prompts; return softmax probs."""
        inputs = self._processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0).cpu().numpy()
        return probs


# ---------------------------------------------------------------------------
# Color classification — pure function, no model needed
# ---------------------------------------------------------------------------

# Minimum OpenCV saturation (0–255) to consider a pixel "chromatic" (not grey/bg).
_MIN_CHROMATIC_SATURATION = 35
# If fewer than this fraction of pixels are chromatic, fall back to median.
_MIN_CHROMATIC_FRACTION = 0.15
# Histogram bins over OpenCV hue 0–180.
_HUE_BINS = 18


def classify_color(crop_bgr: np.ndarray) -> ClothingColor:
    """Return dominant clothing color using a hue histogram on chromatic pixels only."""
    hls = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HLS)
    h_flat = hls[:, :, 0].flatten().astype(float)
    l_flat = hls[:, :, 1].flatten().astype(float)
    s_flat = hls[:, :, 2].flatten().astype(float)

    # Chromatic mask: skip near-black, near-white, and low-saturation pixels.
    chromatic = (
        (s_flat >= _MIN_CHROMATIC_SATURATION)
        & (l_flat >= 40)
        & (l_flat <= 220)
    )

    if chromatic.sum() / max(len(h_flat), 1) >= _MIN_CHROMATIC_FRACTION:
        # Histogram of hues from chromatic pixels — dominant bin wins.
        bin_width = 180.0 / _HUE_BINS
        hist, _ = np.histogram(h_flat[chromatic], bins=_HUE_BINS, range=(0, 180))
        dominant_h = float(np.argmax(hist)) * bin_width + bin_width / 2.0
        dominant_l = float(np.median(l_flat[chromatic]))
        dominant_s = float(np.median(s_flat[chromatic]))
    else:
        # Mostly achromatic crop — use median to distinguish black/white/grey.
        dominant_h = float(np.median(h_flat))
        dominant_l = float(np.median(l_flat))
        dominant_s = float(np.median(s_flat))

    return _hls_to_color(dominant_h, dominant_l, dominant_s)


def _hls_to_color(h_opencv: float, l_val: float, s_val: float) -> ClothingColor:
    """Map OpenCV HLS values to a ClothingColor enum value."""
    # Achromatic checks first (lightness-based, saturation-based).
    if l_val < _BLACK_LIGHTNESS_MAX:
        return ClothingColor.BLACK
    if l_val > _WHITE_LIGHTNESS_MIN:
        return ClothingColor.WHITE
    if s_val < _GREY_SATURATION_MAX:
        return ClothingColor.GREY

    # OpenCV hue is 0–180 (half of 0–360) → scale to degrees.
    hue_degrees = h_opencv * 2.0

    for hue_min, hue_max, color in _HUE_RANGES:
        if hue_min <= hue_degrees < hue_max:
            return color

    return ClothingColor.GREY  # unreachable fallback


# ---------------------------------------------------------------------------
# Crop region helpers
# ---------------------------------------------------------------------------

def _torso_region(crop_bgr: np.ndarray) -> np.ndarray:
    """Slice out the central torso area, skipping face/hair and legs."""
    h, w = crop_bgr.shape[:2]
    y1, y2 = int(h * _TORSO_Y_START), int(h * _TORSO_Y_END)
    x1, x2 = int(w * _TORSO_X_START), int(w * _TORSO_X_END)
    region = crop_bgr[y1:y2, x1:x2]
    return region if region.size > 0 else crop_bgr


def _upper_body_region(crop_bgr: np.ndarray) -> np.ndarray:
    """Slice out the upper body (shirt/jacket area), excluding legs."""
    h, w = crop_bgr.shape[:2]
    y2 = int(h * _UPPER_Y_END)
    region = crop_bgr[:y2, :]
    return region if region.size > 0 else crop_bgr


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _bgr_to_pil(crop_bgr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR array to a PIL RGB image."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_crop(image_path: Path) -> np.ndarray:
    """Read a JPEG crop from disk and return it as a BGR numpy array."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Crop image not found: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    return img
