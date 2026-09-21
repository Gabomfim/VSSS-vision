"""High-cost adaptive ensemble used to create offline pseudo-labels."""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .adaptive_tracker import AdaptiveBallTracker, AdaptiveTrackerConfig
from .fast_tracker import FastBallTracker, FastTrackerConfig
from .tracker import BallDetection


@dataclass(slots=True)
class TeacherConfig:
    target_hsv: tuple[int, int, int]
    radius: int
    hue_tolerance: int = 12
    saturation_tolerance: int = 75
    value_tolerance: int = 75
    radius_tolerance: float = 0.30
    learning_rate: float = 0.08
    minimum_votes: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class PseudoLabel:
    frame_index: int
    center: tuple[float, float]
    radius: float
    confidence: float
    votes: int


class ExpensiveTeacher:
    """Full-frame, multi-radius ensemble with conservative online adaptation."""

    def __init__(self, config: TeacherConfig) -> None:
        self.config = config
        adaptive_config = AdaptiveTrackerConfig(
            radius=config.radius,
            radius_tolerance=config.radius_tolerance,
            hue_tolerance=config.hue_tolerance,
            saturation_tolerance=config.saturation_tolerance,
            value_tolerance=config.value_tolerance,
            learning_rate=config.learning_rate,
            adaptation_min_score=0.72,
        )
        self.model = AdaptiveBallTracker(config.target_hsv, adaptive_config)
        self._previous: np.ndarray | None = None
        self._velocity = np.zeros(2, dtype=float)

    def _candidate_trackers(self) -> list[tuple[str, FastBallTracker]]:
        candidates = []
        for radius_offset in (-2, 0, 2):
            radius = max(3, self.config.radius + radius_offset)
            for detector in ("matched_filter", "components"):
                cfg = FastTrackerConfig(
                    radius=radius,
                    radius_tolerance=self.config.radius_tolerance,
                    hue_tolerance=self.config.hue_tolerance,
                    saturation_tolerance=self.config.saturation_tolerance,
                    value_tolerance=self.config.value_tolerance,
                    detector=detector,
                    preprocessing="open_close",
                    use_roi=False,
                    recovery_scale=1.0,
                    minimum_score=0.42,
                    matched_filter_threshold=0.28,
                )
                candidates.append((f"{detector}:{radius}", FastBallTracker(self.model.target_hsv, cfg)))
        return candidates

    def _hough_candidate(self, frame: np.ndarray) -> BallDetection | None:
        mask = self.model.color_mask(frame)
        blurred = cv2.GaussianBlur(mask, (9, 9), 1.5)
        r = self.config.radius
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max(4, r * 2),
            param1=80,
            param2=8,
            minRadius=max(2, r - 3),
            maxRadius=r + 3,
        )
        if circles is None:
            return None
        options = circles[0]
        if self._previous is not None:
            predicted = self._previous + self._velocity
            choice = min(options, key=lambda circle: np.linalg.norm(circle[:2] - predicted))
        else:
            choice = options[0]
        return BallDetection((round(float(choice[0])), round(float(choice[1]))), float(choice[2]), 0.75)

    def detect(self, frame: np.ndarray, frame_index: int) -> PseudoLabel | None:
        proposals: list[BallDetection] = []
        for _, tracker in self._candidate_trackers():
            result = tracker.detect_timed(frame, update=False).detection
            if result is not None:
                proposals.append(result)
        hough = self._hough_candidate(frame)
        if hough is not None:
            proposals.append(hough)
        if not proposals:
            return None

        predicted = self._previous + self._velocity if self._previous is not None else None
        clusters: list[list[BallDetection]] = []
        for proposal in proposals:
            for cluster in clusters:
                centroid = np.mean([item.center for item in cluster], axis=0)
                if np.linalg.norm(np.asarray(proposal.center) - centroid) <= self.config.radius * 0.8:
                    cluster.append(proposal)
                    break
            else:
                clusters.append([proposal])

        def cluster_rank(cluster: list[BallDetection]) -> float:
            confidence = sum(item.score for item in cluster)
            if predicted is not None:
                center = np.mean([item.center for item in cluster], axis=0)
                confidence -= 0.25 * np.linalg.norm(center - predicted) / max(1, self.config.radius)
            return confidence

        winning = max(clusters, key=cluster_rank)
        if len(winning) < self.config.minimum_votes:
            return None
        weights = np.asarray([max(0.05, item.score) for item in winning])
        centers = np.asarray([item.center for item in winning], dtype=float)
        center_array = np.average(centers, axis=0, weights=weights)
        radius = float(np.average([item.radius for item in winning], weights=weights))
        spread = float(np.mean(np.linalg.norm(centers - center_array, axis=1)))
        agreement = max(0.0, 1.0 - spread / max(1, self.config.radius))
        confidence = min(1.0, 0.55 * agreement + 0.45 * min(1.0, len(winning) / 5))
        center = np.rint(center_array).astype(int)
        detection = BallDetection((int(center[0]), int(center[1])), radius, confidence)

        if confidence >= 0.68:
            if self._previous is not None:
                measured_velocity = center_array - self._previous
                self._velocity = 0.65 * self._velocity + 0.35 * measured_velocity
            self._previous = center_array
            self.model.update_color_model(frame, detection)
        return PseudoLabel(frame_index, tuple(center_array), radius, confidence, len(winning))

    def label(self, frames: list[np.ndarray]) -> list[PseudoLabel | None]:
        return [self.detect(frame, index) for index, frame in enumerate(frames)]
