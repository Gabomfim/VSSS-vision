"""Agreement and latency metrics for fast-tracker versus robust teacher evaluation."""

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np

from .fast_tracker import TimedDetection
from .teacher import PseudoLabel


@dataclass(slots=True)
class EvaluationAccumulator:
    ball_radius: float
    processed_frames: int = 0
    source_frames_skipped: int = 0
    teacher_detections: int = 0
    fast_detections: int = 0
    matched_detections: int = 0
    fast_misses: int = 0
    fast_only_detections: int = 0
    center_errors: list[float] = field(default_factory=list)
    fast_latencies_ms: list[float] = field(default_factory=list)
    teacher_latencies_ms: list[float] = field(default_factory=list)
    teacher_confidences: list[float] = field(default_factory=list)

    def update(
        self,
        fast: TimedDetection,
        teacher: PseudoLabel | None,
        teacher_latency_ms: float,
        skipped_source_frames: int = 0,
    ) -> float | None:
        self.processed_frames += 1
        self.source_frames_skipped += max(0, skipped_source_frames)
        self.fast_latencies_ms.append(fast.total_ms)
        self.teacher_latencies_ms.append(teacher_latency_ms)
        if fast.detection is not None:
            self.fast_detections += 1
        if teacher is not None:
            self.teacher_detections += 1
            self.teacher_confidences.append(teacher.confidence)
        if fast.detection is None and teacher is not None:
            self.fast_misses += 1
            return None
        if fast.detection is not None and teacher is None:
            self.fast_only_detections += 1
            return None
        if fast.detection is None or teacher is None:
            return None
        error = float(
            np.linalg.norm(
                np.asarray(fast.detection.center, dtype=float)
                - np.asarray(teacher.center, dtype=float)
            )
        )
        self.matched_detections += 1
        self.center_errors.append(error)
        return error

    @staticmethod
    def _distribution(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "median": None, "p95": None, "maximum": None}
        array = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "maximum": float(np.max(array)),
        }

    def summary(self) -> dict:
        errors = np.asarray(self.center_errors, dtype=float)
        teacher_count = max(1, self.teacher_detections)
        processed = max(1, self.processed_frames)
        return {
            "interpretation": (
                "Agreement with the expensive teacher is a pseudo-label metric, not independent ground truth. "
                "Validate systematic bias with manually labeled frames."
            ),
            "ball_radius_px": self.ball_radius,
            "processed_frames": self.processed_frames,
            "source_frames_skipped": self.source_frames_skipped,
            "teacher_detection_rate": self.teacher_detections / processed,
            "fast_detection_rate": self.fast_detections / processed,
            "fast_recall_against_teacher": self.matched_detections / teacher_count,
            "fast_miss_rate_against_teacher": self.fast_misses / teacher_count,
            "fast_only_detection_rate": self.fast_only_detections / processed,
            "matched_detections": self.matched_detections,
            "center_error_px": self._distribution(self.center_errors),
            "agreement_within_quarter_radius": (
                float(np.mean(errors <= self.ball_radius * 0.25)) if errors.size else None
            ),
            "agreement_within_half_radius": (
                float(np.mean(errors <= self.ball_radius * 0.5)) if errors.size else None
            ),
            "fast_latency_ms": self._distribution(self.fast_latencies_ms),
            "teacher_latency_ms": self._distribution(self.teacher_latencies_ms),
            "teacher_confidence": self._distribution(self.teacher_confidences),
        }

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        temporary.replace(destination)
