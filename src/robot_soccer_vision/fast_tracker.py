"""Low-latency adaptive tracker with ROI prediction and fixed circle kernels."""

from dataclasses import asdict, dataclass
from time import perf_counter_ns

import cv2
import numpy as np

from .adaptive_tracker import AdaptiveBallTracker, AdaptiveTrackerConfig
from .tracker import BallDetection


@dataclass(slots=True)
class FastTrackerConfig(AdaptiveTrackerConfig):
    preprocessing: str = "open"
    detector: str = "matched_filter"
    use_roi: bool = True
    roi_radius_multiplier: float = 5.0
    maximum_roi_misses: int = 2
    recovery_scale: float = 0.5
    matched_filter_threshold: float = 0.34

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class TimedDetection:
    detection: BallDetection | None
    total_ms: float
    preprocess_ms: float
    detect_ms: float
    used_roi: bool
    refinement_ms: float = 0.0


class FastBallTracker(AdaptiveBallTracker):
    """Cheap online tracker; expensive global search is only a recovery path."""

    config: FastTrackerConfig

    def __init__(
        self,
        target_hsv: tuple[int, int, int] = (15, 220, 220),
        config: FastTrackerConfig | None = None,
    ) -> None:
        super().__init__(target_hsv, config or FastTrackerConfig())
        self._last_center: np.ndarray | None = None
        self._misses = 0
        self._kernel_radius = -1
        self._kernel: np.ndarray | None = None
        self._kalman_state: np.ndarray | None = None
        self._kalman_covariance = np.eye(4, dtype=float)
        self._transition = np.asarray(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        self._observation = np.asarray(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float
        )
        self._process_noise = np.diag([0.2, 0.2, 1.0, 1.0])
        self._measurement_noise = np.eye(2, dtype=float) * 2.0
        self.set_radius(self.config.radius)

    def reset_motion(self) -> None:
        self._last_center = None
        self._misses = 0
        self._kalman_state = None
        self._kalman_covariance = np.eye(4, dtype=float)

    def set_radius(self, radius: int) -> None:
        """Update the selected radius and eagerly precompute its fixed kernel."""
        radius = max(2, int(radius))
        self.config.radius = radius
        self._circle_kernel(radius)

    def _kalman_predict(self) -> np.ndarray | None:
        if self._kalman_state is None:
            return None
        self._kalman_state = self._transition @ self._kalman_state
        self._kalman_covariance = (
            self._transition @ self._kalman_covariance @ self._transition.T
            + self._process_noise
        )
        return self._kalman_state[:2].copy()

    def _kalman_correct(self, center: np.ndarray) -> None:
        if self._kalman_state is None:
            self._kalman_state = np.asarray([center[0], center[1], 0.0, 0.0])
            self._kalman_covariance = np.diag([2.0, 2.0, 16.0, 16.0])
            return
        innovation_covariance = (
            self._observation @ self._kalman_covariance @ self._observation.T
            + self._measurement_noise
        )
        gain = (
            self._kalman_covariance
            @ self._observation.T
            @ np.linalg.pinv(innovation_covariance)
        )
        innovation = center - self._observation @ self._kalman_state
        self._kalman_state += gain @ innovation
        self._kalman_covariance = (
            np.eye(4) - gain @ self._observation
        ) @ self._kalman_covariance

    def _circle_kernel(self, radius: int) -> np.ndarray:
        if self._kernel is not None and radius == self._kernel_radius:
            return self._kernel
        outer = max(radius + 2, round(radius * 1.65))
        size = outer * 2 + 1
        yy, xx = np.ogrid[-outer : outer + 1, -outer : outer + 1]
        distance = np.sqrt(xx * xx + yy * yy)
        inside = distance <= radius
        ring = (distance > radius * 1.15) & (distance <= outer)
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[inside] = 1.0 / max(1, np.count_nonzero(inside))
        kernel[ring] = -1.0 / max(1, np.count_nonzero(ring))
        self._kernel_radius = radius
        self._kernel = kernel
        return kernel

    def _predicted_roi(
        self, shape: tuple[int, ...], predicted: np.ndarray | None
    ) -> tuple[int, int, int, int] | None:
        if not self.config.use_roi or predicted is None:
            return None
        speed = 0.0 if self._kalman_state is None else np.linalg.norm(self._kalman_state[2:])
        half = int(
            max(
                self.config.radius * self.config.roi_radius_multiplier,
                self.config.radius * 3 + speed * 1.5,
            )
        )
        height, width = shape[:2]
        x0, y0 = np.floor(predicted - half).astype(int)
        x1, y1 = np.ceil(predicted + half + 1).astype(int)
        return max(0, x0), max(0, y0), min(width, x1), min(height, y1)

    def _preprocess_mask(self, frame: np.ndarray) -> np.ndarray:
        mask = self.raw_color_mask(frame)
        radius = max(2, self.config.radius)
        if self.config.preprocessing == "none":
            return mask
        if self.config.preprocessing == "gaussian":
            return cv2.GaussianBlur(mask, (5, 5), 0)
        kernel_size = max(3, (radius // 4) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if self.config.preprocessing == "open_close":
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _matched_detection(self, mask: np.ndarray) -> BallDetection | None:
        response = cv2.filter2D(
            mask.astype(np.float32) / 255.0,
            cv2.CV_32F,
            self._circle_kernel(max(2, self.config.radius)),
            borderType=cv2.BORDER_CONSTANT,
        )
        _, peak, _, location = cv2.minMaxLoc(response)
        if peak < self.config.matched_filter_threshold:
            return None
        return BallDetection(location, float(self.config.radius), float(min(1.0, peak)))

    def _component_detection(self, mask: np.ndarray) -> BallDetection | None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        expected = self.config.radius
        lower = expected * (1 - self.config.radius_tolerance)
        upper = expected * (1 + self.config.radius_tolerance)
        best = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 0:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if not lower <= radius <= upper:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / max(perimeter * perimeter, 1.0)
            radius_score = 1 - abs(radius - expected) / expected
            score = 0.7 * circularity + 0.3 * radius_score
            candidate = BallDetection((round(x), round(y)), radius, float(score))
            if best is None or candidate.score > best.score:
                best = candidate
        return best

    def detect_timed(self, frame_bgr: np.ndarray, update: bool = True) -> TimedDetection:
        started = perf_counter_ns()
        predicted = self._kalman_predict()
        roi = self._predicted_roi(frame_bgr.shape, predicted)
        used_roi = roi is not None and self._misses <= self.config.maximum_roi_misses
        if used_roi:
            x0, y0, x1, y1 = roi
            search = frame_bgr[y0:y1, x0:x1]
        else:
            x0 = y0 = 0
            search = frame_bgr

        search_scale = 1.0
        original_radius = self.config.radius
        if not used_roi and 0.0 < self.config.recovery_scale < 1.0:
            search_scale = self.config.recovery_scale
            search = cv2.resize(
                search, None, fx=search_scale, fy=search_scale, interpolation=cv2.INTER_AREA
            )
            self.config.radius = max(2, round(original_radius * search_scale))

        try:
            preprocess_started = perf_counter_ns()
            mask = self._preprocess_mask(search)
            preprocess_ms = (perf_counter_ns() - preprocess_started) / 1e6
            detect_started = perf_counter_ns()
            if self.config.detector == "components":
                detection = self._component_detection(mask)
            else:
                detection = self._matched_detection(mask)
            detect_ms = (perf_counter_ns() - detect_started) / 1e6
        finally:
            self.config.radius = original_radius

        refinement_ms = 0.0
        if detection is not None:
            global_center = (
                round(detection.center[0] / search_scale) + x0,
                round(detection.center[1] / search_scale) + y0,
            )
            detection = BallDetection(
                global_center, detection.radius / search_scale, detection.score
            )
            if search_scale < 1.0:
                refinement_started = perf_counter_ns()
                half = max(original_radius * 3, 12)
                height, width = frame_bgr.shape[:2]
                rx0 = max(0, global_center[0] - half)
                ry0 = max(0, global_center[1] - half)
                rx1 = min(width, global_center[0] + half + 1)
                ry1 = min(height, global_center[1] + half + 1)
                native_search = frame_bgr[ry0:ry1, rx0:rx1]
                native_mask = self._preprocess_mask(native_search)
                native_detection = (
                    self._component_detection(native_mask)
                    if self.config.detector == "components"
                    else self._matched_detection(native_mask)
                )
                if native_detection is not None:
                    detection = BallDetection(
                        (
                            native_detection.center[0] + rx0,
                            native_detection.center[1] + ry0,
                        ),
                        native_detection.radius,
                        native_detection.score,
                    )
                refinement_ms = (perf_counter_ns() - refinement_started) / 1e6
            center = np.asarray(global_center, dtype=float)
            center = np.asarray(detection.center, dtype=float)
            self._last_center = center
            self._kalman_correct(center)
            self._misses = 0
            if update and detection.score >= self.config.adaptation_min_score:
                self.update_color_model(frame_bgr, detection)
        else:
            self._misses += 1

        return TimedDetection(
            detection,
            (perf_counter_ns() - started) / 1e6,
            preprocess_ms,
            detect_ms,
            used_roi,
            refinement_ms,
        )
