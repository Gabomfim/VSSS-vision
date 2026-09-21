"""Online color adaptation for changing field illumination."""

from dataclasses import dataclass

import cv2
import numpy as np

from .tracker import BallDetection, BallTracker, TrackerConfig


@dataclass(slots=True)
class AdaptiveTrackerConfig(TrackerConfig):
    learning_rate: float = 0.12
    adaptation_min_score: float = 0.72
    maximum_hue_step: float = 2.5
    minimum_saturation: int = 45
    sample_radius_ratio: float = 0.65


def _signed_hue_delta(new_hue: float, old_hue: float) -> float:
    """Shortest signed distance between OpenCV hues on the 0..179 circle."""
    return (new_hue - old_hue + 90.0) % 180.0 - 90.0


class AdaptiveBallTracker(BallTracker):
    """Track the ball and slowly learn its appearance from trusted detections.

    Shape and radius are kept fixed. Only the HSV color model changes, and only
    after a high-confidence circle match, which limits background-driven drift.
    """

    config: AdaptiveTrackerConfig

    def __init__(
        self,
        target_hsv: tuple[int, int, int] = (15, 220, 220),
        config: AdaptiveTrackerConfig | None = None,
    ) -> None:
        super().__init__(target_hsv, config or AdaptiveTrackerConfig())
        self.adaptation_enabled = True
        self.update_count = 0

    def detect_and_update(self, frame_bgr: np.ndarray) -> BallDetection | None:
        detection = self.detect(frame_bgr)
        if (
            detection is not None
            and self.adaptation_enabled
            and detection.score >= self.config.adaptation_min_score
        ):
            self.update_color_model(frame_bgr, detection)
        return detection

    def update_color_model(
        self, frame_bgr: np.ndarray, detection: BallDetection
    ) -> tuple[int, int, int]:
        """Apply a guarded exponential moving average from the detected ball."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        sample_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        sample_radius = max(2, round(detection.radius * self.config.sample_radius_ratio))
        cv2.circle(sample_mask, detection.center, sample_radius, 255, -1)
        pixels = hsv[sample_mask > 0]

        old = np.asarray(self.target_hsv, dtype=np.float64)
        hue_distance = np.abs(pixels[:, 0].astype(float) - old[0])
        hue_distance = np.minimum(hue_distance, 180.0 - hue_distance)
        trustworthy = pixels[
            (hue_distance <= max(18, self.config.hue_tolerance * 2))
            & (pixels[:, 1] >= self.config.minimum_saturation)
        ]
        if trustworthy.shape[0] < 8:
            return self.target_hsv

        observed = np.median(trustworthy, axis=0).astype(float)
        rate = float(np.clip(self.config.learning_rate, 0.0, 1.0))
        hue_step = np.clip(
            _signed_hue_delta(observed[0], old[0]) * rate,
            -self.config.maximum_hue_step,
            self.config.maximum_hue_step,
        )
        updated = np.array(
            [
                (old[0] + hue_step) % 180.0,
                old[1] + rate * (observed[1] - old[1]),
                old[2] + rate * (observed[2] - old[2]),
            ]
        )
        self.target_hsv = tuple(int(round(value)) for value in updated)
        self.update_count += 1
        return self.target_hsv

