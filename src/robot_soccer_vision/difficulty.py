"""Difficulty scoring and random active-learning sample selection."""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .fast_tracker import FastBallTracker, FastTrackerConfig
from .teacher import ExpensiveTeacher, TeacherConfig
from .field_rectifier import FieldRectifier


@dataclass(slots=True)
class DifficultyRecord:
    frame_index: int
    difficulty: float = 0.0
    teacher_x: float | None = None
    teacher_y: float | None = None
    teacher_radius: float | None = None
    teacher_confidence: float = 0.0
    teacher_votes: int = 0
    student_teacher_distance: float | None = None
    temporal_surprise: float = 0.0
    blur_score: float = 0.0
    median_brightness: float = 0.0
    shadow_fraction: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def configurations_from_report(report: dict) -> tuple[TeacherConfig, FastTrackerConfig]:
    item = report["teacher"]
    teacher = TeacherConfig(
        target_hsv=tuple(item["target_hsv"]),
        radius=int(item["radius"]),
        hue_tolerance=int(item["hue_tolerance"]),
        saturation_tolerance=int(item["saturation_tolerance"]),
        value_tolerance=int(item["value_tolerance"]),
        radius_tolerance=float(item["radius_tolerance"]),
        learning_rate=float(item["learning_rate"]),
        minimum_votes=int(item.get("minimum_votes", 2)),
    )
    selected = report["recommended_student"]
    student = FastTrackerConfig(
        radius=teacher.radius,
        radius_tolerance=teacher.radius_tolerance,
        hue_tolerance=teacher.hue_tolerance,
        saturation_tolerance=teacher.saturation_tolerance,
        value_tolerance=teacher.value_tolerance,
        learning_rate=teacher.learning_rate,
        preprocessing=selected["preprocessing"],
        detector=selected["detector"],
        use_roi=bool(selected["use_roi"]),
    )
    return teacher, student


def rectifier_from_report(report: dict) -> FieldRectifier | None:
    calibration = report.get("field_calibration")
    if not calibration:
        return None
    rectifier = FieldRectifier()
    for point in calibration["corners"]:
        rectifier.add_point((round(point[0]), round(point[1])))
    return rectifier


def _visual_quality(frame: np.ndarray) -> tuple[float, float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    median_brightness = float(np.median(gray))
    shadow_fraction = float(np.mean(gray < 30))
    return blur_score, median_brightness, shadow_fraction


def score_video_difficulty(
    video_path: str,
    teacher_config: TeacherConfig,
    student_config: FastTrackerConfig,
    maximum_frames: int | None = None,
    stride: int = 1,
    rectifier: FieldRectifier | None = None,
) -> list[DifficultyRecord]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    teacher = ExpensiveTeacher(teacher_config)
    student = FastBallTracker(teacher_config.target_hsv, student_config)
    records: list[DifficultyRecord] = []
    previous_center = None
    previous_velocity = np.zeros(2, dtype=float)
    frame_index = 0
    try:
        while maximum_frames is None or frame_index < maximum_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % max(1, stride) != 0:
                frame_index += 1
                continue
            if rectifier is not None:
                frame = rectifier.warp(frame)
            label = teacher.detect(frame, frame_index)
            student_result = student.detect_timed(frame).detection
            blur, brightness, shadows = _visual_quality(frame)
            distance = None
            surprise = 0.0
            if label is not None:
                center = np.asarray(label.center)
                if student_result is not None:
                    distance = float(
                        np.linalg.norm(center - np.asarray(student_result.center))
                    )
                if previous_center is not None:
                    predicted = previous_center + previous_velocity
                    surprise = float(np.linalg.norm(center - predicted))
                    measured_velocity = center - previous_center
                    previous_velocity = 0.65 * previous_velocity + 0.35 * measured_velocity
                previous_center = center
            records.append(
                DifficultyRecord(
                    frame_index=frame_index,
                    teacher_x=None if label is None else label.center[0],
                    teacher_y=None if label is None else label.center[1],
                    teacher_radius=None if label is None else label.radius,
                    teacher_confidence=0.0 if label is None else label.confidence,
                    teacher_votes=0 if label is None else label.votes,
                    student_teacher_distance=distance,
                    temporal_surprise=surprise,
                    blur_score=blur,
                    median_brightness=brightness,
                    shadow_fraction=shadows,
                )
            )
            frame_index += 1
    finally:
        capture.release()
    _assign_composite_difficulty(records, teacher_config.radius)
    return records


def _robust_unit(values: np.ndarray, invert: bool = False) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.ones_like(values, dtype=float)
    if np.any(finite):
        low, high = np.percentile(values[finite], [10, 90])
        if high <= low:
            result[finite] = 0.0
        else:
            result[finite] = np.clip((values[finite] - low) / (high - low), 0, 1)
        if invert:
            result[finite] = 1.0 - result[finite]
    return result


def _assign_composite_difficulty(records: list[DifficultyRecord], radius: int) -> None:
    if not records:
        return
    confidence = np.asarray([item.teacher_confidence for item in records])
    disagreement = np.asarray(
        [
            np.nan if item.student_teacher_distance is None else item.student_teacher_distance
            for item in records
        ]
    )
    temporal = np.asarray([item.temporal_surprise for item in records])
    blur = np.asarray([item.blur_score for item in records])
    brightness = np.asarray([item.median_brightness for item in records])
    shadows = np.asarray([item.shadow_fraction for item in records])
    no_teacher = np.asarray([item.teacher_x is None for item in records], dtype=float)
    no_student_agreement = np.asarray(
        [item.teacher_x is not None and item.student_teacher_distance is None for item in records],
        dtype=float,
    )
    difficulty = (
        0.25 * (1.0 - confidence)
        + 0.22 * np.nan_to_num(disagreement / max(1, radius), nan=1.0, posinf=1.0)
        + 0.14 * _robust_unit(temporal)
        + 0.14 * _robust_unit(blur, invert=True)
        + 0.10 * _robust_unit(brightness, invert=True)
        + 0.08 * _robust_unit(shadows)
        + 0.05 * no_teacher
        + 0.02 * no_student_agreement
    )
    difficulty = np.clip(difficulty, 0, 1)
    for item, score in zip(records, difficulty):
        item.difficulty = float(score)


def sample_difficult_frames(
    records: list[DifficultyRecord],
    count: int,
    pool_fraction: float = 0.25,
    seed: int = 7,
    excluded_indices: set[int] | None = None,
) -> list[DifficultyRecord]:
    excluded = excluded_indices or set()
    available = [item for item in records if item.frame_index not in excluded]
    if not available or count <= 0:
        return []
    pool_size = max(count, int(np.ceil(len(available) * np.clip(pool_fraction, 0.01, 1.0))))
    pool = sorted(available, key=lambda item: item.difficulty, reverse=True)[:pool_size]
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(pool), size=min(count, len(pool)), replace=False)
    return sorted((pool[int(index)] for index in selected_indices), key=lambda item: item.frame_index)


def load_selected_frames(
    video_path: str,
    selected: list[DifficultyRecord],
    rectifier: FieldRectifier | None = None,
) -> dict[int, np.ndarray]:
    wanted = {item.frame_index for item in selected}
    capture = cv2.VideoCapture(video_path)
    frames = {}
    index = 0
    try:
        while wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                frames[index] = rectifier.warp(frame) if rectifier is not None else frame
                wanted.remove(index)
            index += 1
    finally:
        capture.release()
    return frames
