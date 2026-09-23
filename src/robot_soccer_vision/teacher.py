"""High-cost adaptive ensemble used to create offline pseudo-labels."""

from collections.abc import Callable
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

    def label_bidirectional(
        self,
        frames: list[np.ndarray],
        maximum_gap: int = 4,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> list[PseudoLabel | None]:
        """Label recorded video using past and future evidence, then smooth offline."""
        total = max(1, len(frames) * 2)
        forward_teacher = ExpensiveTeacher(self.config)
        forward = []
        for index, frame in enumerate(frames):
            forward.append(forward_teacher.detect(frame, index))
            if progress is not None:
                progress(index + 1, total, "forward pass")
        backward_teacher = ExpensiveTeacher(self.config)
        backward_raw = []
        for index, frame in enumerate(reversed(frames)):
            backward_raw.append(backward_teacher.detect(frame, index))
            if progress is not None:
                progress(len(frames) + index + 1, total, "backward pass")
        backward = [
            None
            if label is None
            else PseudoLabel(
                len(frames) - 1 - label.frame_index,
                label.center,
                label.radius,
                label.confidence,
                label.votes,
            )
            for label in reversed(backward_raw)
        ]
        labels, diagnostics = fuse_bidirectional_labels(
            forward, backward, self.config.radius, maximum_gap
        )
        self.bidirectional_diagnostics = diagnostics
        return labels


def _combined_label(
    index: int,
    first: PseudoLabel,
    second: PseudoLabel,
    expected_radius: float,
) -> PseudoLabel | None:
    distance = float(np.linalg.norm(np.asarray(first.center) - second.center))
    association_limit = max(3.0, expected_radius * 1.5)
    if distance > association_limit:
        return None
    weights = np.asarray([max(0.05, first.confidence), max(0.05, second.confidence)])
    center = np.average(np.asarray([first.center, second.center]), axis=0, weights=weights)
    radius = float(np.average([first.radius, second.radius], weights=weights))
    agreement = max(0.0, 1.0 - distance / association_limit)
    confidence = min(
        1.0, 0.55 * max(first.confidence, second.confidence) + 0.45 * agreement
    )
    return PseudoLabel(index, tuple(center), radius, confidence, first.votes + second.votes)


def _fill_supported_gaps(
    labels: list[PseudoLabel | None], expected_radius: float, maximum_gap: int
) -> int:
    filled = 0
    index = 0
    while index < len(labels):
        if labels[index] is not None:
            index += 1
            continue
        start = index
        while index < len(labels) and labels[index] is None:
            index += 1
        gap = index - start
        if start == 0 or index == len(labels) or gap > maximum_gap:
            continue
        left, right = labels[start - 1], labels[index]
        if left is None or right is None:
            continue
        displacement = float(np.linalg.norm(np.asarray(right.center) - left.center))
        if displacement > expected_radius * 2.5 * (gap + 1):
            continue
        for offset in range(1, gap + 1):
            fraction = offset / (gap + 1)
            center = (
                (1 - fraction) * np.asarray(left.center)
                + fraction * np.asarray(right.center)
            )
            radius = (1 - fraction) * left.radius + fraction * right.radius
            confidence = 0.70 * min(left.confidence, right.confidence)
            labels[start + offset - 1] = PseudoLabel(
                start + offset - 1,
                tuple(center),
                float(radius),
                confidence,
                min(left.votes, right.votes),
            )
            filled += 1
    return filled


def _rts_smooth_segment(
    labels: list[PseudoLabel], expected_radius: float
) -> list[PseudoLabel]:
    """Constant-velocity Kalman filter followed by a Rauch-Tung-Striebel pass."""
    count = len(labels)
    if count < 3:
        return labels
    transition = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    observation = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    process_noise = np.diag([0.15, 0.15, 0.8, 0.8]) * max(1.0, expected_radius / 10)
    states = np.zeros((count, 4), dtype=float)
    covariances = np.zeros((count, 4, 4), dtype=float)
    predicted_states = np.zeros_like(states)
    predicted_covariances = np.zeros_like(covariances)
    states[0, :2] = labels[0].center
    covariances[0] = np.diag([4.0, 4.0, 25.0, 25.0])
    identity = np.eye(4)
    for index in range(1, count):
        predicted_states[index] = transition @ states[index - 1]
        predicted_covariances[index] = (
            transition @ covariances[index - 1] @ transition.T + process_noise
        )
        measurement = np.asarray(labels[index].center)
        variance = max(0.8, expected_radius * (1.05 - labels[index].confidence)) ** 2
        measurement_noise = np.eye(2) * variance
        innovation_covariance = (
            observation @ predicted_covariances[index] @ observation.T + measurement_noise
        )
        gain = (
            predicted_covariances[index]
            @ observation.T
            @ np.linalg.pinv(innovation_covariance)
        )
        states[index] = predicted_states[index] + gain @ (
            measurement - observation @ predicted_states[index]
        )
        covariances[index] = (identity - gain @ observation) @ predicted_covariances[index]
    smoothed = states.copy()
    smoothed_covariances = covariances.copy()
    for index in range(count - 2, -1, -1):
        gain = (
            covariances[index]
            @ transition.T
            @ np.linalg.pinv(predicted_covariances[index + 1])
        )
        smoothed[index] += gain @ (smoothed[index + 1] - predicted_states[index + 1])
        smoothed_covariances[index] += gain @ (
            smoothed_covariances[index + 1] - predicted_covariances[index + 1]
        ) @ gain.T
    return [
        PseudoLabel(
            label.frame_index,
            tuple(smoothed[index, :2]),
            label.radius,
            label.confidence,
            label.votes,
        )
        for index, label in enumerate(labels)
    ]


def fuse_bidirectional_labels(
    forward: list[PseudoLabel | None],
    backward: list[PseudoLabel | None],
    expected_radius: float,
    maximum_gap: int = 4,
) -> tuple[list[PseudoLabel | None], dict]:
    """Associate both temporal directions and produce conservative offline labels."""
    if len(forward) != len(backward):
        raise ValueError("Forward and backward label sequences must have the same length")
    fused: list[PseudoLabel | None] = []
    paired = rejected = unilateral = 0
    for index, (left, right) in enumerate(zip(forward, backward)):
        if left is not None and right is not None:
            label = _combined_label(index, left, right, expected_radius)
            paired += label is not None
            rejected += label is None
            fused.append(label)
        elif left is not None or right is not None:
            candidate = left if left is not None else right
            if candidate is not None and candidate.confidence >= 0.82 and candidate.votes >= 3:
                fused.append(
                    PseudoLabel(
                        index,
                        candidate.center,
                        candidate.radius,
                        candidate.confidence * 0.8,
                        candidate.votes,
                    )
                )
                unilateral += 1
            else:
                fused.append(None)
        else:
            fused.append(None)
    interpolated = _fill_supported_gaps(fused, expected_radius, maximum_gap)
    index = 0
    while index < len(fused):
        if fused[index] is None:
            index += 1
            continue
        end = index
        segment: list[PseudoLabel] = []
        while end < len(fused) and fused[end] is not None:
            item = fused[end]
            assert item is not None
            segment.append(item)
            end += 1
        smoothed = _rts_smooth_segment(segment, expected_radius)
        fused[index:end] = smoothed
        index = end
    diagnostics = {
        "mode": "bidirectional_rts",
        "paired_frames": paired,
        "rejected_direction_disagreements": rejected,
        "high_confidence_unilateral_frames": unilateral,
        "interpolated_short_gap_frames": interpolated,
        "final_label_count": sum(label is not None for label in fused),
        "maximum_interpolated_gap": maximum_gap,
    }
    return fused, diagnostics
