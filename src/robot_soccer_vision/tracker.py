"""Color and circle-template ball tracking primitives."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class TrackerConfig:
    radius: int = 12
    radius_tolerance: float = 0.35
    hue_tolerance: int = 12
    saturation_tolerance: int = 90
    value_tolerance: int = 90
    minimum_score: float = 0.45


@dataclass(slots=True, frozen=True)
class BallDetection:
    center: tuple[int, int]
    radius: float
    score: float


def sample_hsv_color(
    frame_bgr: np.ndarray, center: tuple[int, int], radius: int
) -> tuple[int, int, int]:
    """Return a robust HSV sample from the circular calibration overlay."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, max(2, int(radius * 0.65)), 255, -1)
    pixels = hsv[mask > 0]
    if pixels.size == 0:
        raise ValueError("The calibration circle is outside the image")

    # Hue is circular. Taking the median is reliable for the small orange range;
    # S and V medians reject white glare on a glossy ball.
    median = np.median(pixels, axis=0).astype(int)
    return int(median[0]), int(median[1]), int(median[2])


class BallTracker:
    """Find a nearly fixed-size colored circle in an overhead camera frame."""

    def __init__(
        self,
        target_hsv: tuple[int, int, int] = (15, 220, 220),
        config: TrackerConfig | None = None,
    ) -> None:
        self.target_hsv = target_hsv
        self.config = config or TrackerConfig()

    def color_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        target = np.array(self.target_hsv, dtype=np.int16)
        pixels = hsv.astype(np.int16)

        hue_distance = np.abs(pixels[:, :, 0] - target[0])
        hue_distance = np.minimum(hue_distance, 180 - hue_distance)
        selected = (
            (hue_distance <= self.config.hue_tolerance)
            & (np.abs(pixels[:, :, 1] - target[1]) <= self.config.saturation_tolerance)
            & (np.abs(pixels[:, :, 2] - target[2]) <= self.config.value_tolerance)
        )
        mask = (selected.astype(np.uint8) * 255)
        kernel_size = max(3, (self.config.radius // 4) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def detect(self, frame_bgr: np.ndarray) -> BallDetection | None:
        mask = self.color_mask(frame_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        expected = max(2, self.config.radius)
        lower = expected * (1.0 - self.config.radius_tolerance)
        upper = expected * (1.0 + self.config.radius_tolerance)
        best: BallDetection | None = None

        for contour in contours:
            if cv2.contourArea(contour) < np.pi * lower * lower * 0.35:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if not lower <= radius <= upper:
                continue

            # Compare the segmented shape with an ideal filled-circle template.
            x0 = max(0, int(x - radius - 2))
            y0 = max(0, int(y - radius - 2))
            x1 = min(mask.shape[1], int(x + radius + 3))
            y1 = min(mask.shape[0], int(y + radius + 3))
            observed = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            shifted = contour - np.array([[[x0, y0]]], dtype=contour.dtype)
            cv2.drawContours(observed, [shifted], -1, 255, -1)
            template = np.zeros_like(observed)
            cv2.circle(template, (round(x) - x0, round(y) - y0), round(radius), 255, -1)
            intersection = np.count_nonzero((observed > 0) & (template > 0))
            union = np.count_nonzero((observed > 0) | (template > 0))
            template_score = intersection / union if union else 0.0
            radius_score = 1.0 - abs(radius - expected) / max(expected, 1)
            score = 0.75 * template_score + 0.25 * max(0.0, radius_score)

            candidate = BallDetection((round(x), round(y)), radius, score)
            if score >= self.config.minimum_score and (best is None or score > best.score):
                best = candidate
        return best

