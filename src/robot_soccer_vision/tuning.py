"""Self-supervised teacher selection and latency/accuracy ablations."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
import numpy as np

from .fast_tracker import FastBallTracker, FastTrackerConfig
from .teacher import ExpensiveTeacher, PseudoLabel, TeacherConfig
from .tracker import sample_hsv_color
from .field_rectifier import FieldRectifier


@dataclass(slots=True)
class AblationResult:
    preprocessing: str
    detector: str
    use_roi: bool
    mean_center_error_px: float
    p95_center_error_px: float
    recall: float
    mean_latency_ms: float
    p95_latency_ms: float
    objective: float


def load_video_frames(
    path: str,
    maximum_frames: int = 600,
    stride: int = 1,
    rectifier: FieldRectifier | None = None,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames = []
    source_index = 0
    try:
        while len(frames) < maximum_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if source_index % stride == 0:
                frames.append(rectifier.warp(frame) if rectifier is not None else frame)
            source_index += 1
    finally:
        capture.release()
    if not frames:
        raise RuntimeError("The video did not contain readable frames")
    return frames


def _lighting_perturb(frame: np.ndarray, index: int) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    scale = (0.78, 1.18, 0.90, 1.08)[index % 4]
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * scale, 0, 255)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (2.0 - scale * 0.85), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _paired_distance(
    first: list[PseudoLabel | None], second: list[PseudoLabel | None]
) -> tuple[float, float]:
    distances = []
    for left, right in zip(first, second):
        if left is not None and right is not None:
            distances.append(float(np.linalg.norm(np.asarray(left.center) - right.center)))
    overlap = len(distances) / max(1, len(first))
    return (mean(distances) if distances else float("inf")), overlap


def teacher_self_supervision_score(
    frames: list[np.ndarray], config: TeacherConfig
) -> tuple[float, dict]:
    forward = ExpensiveTeacher(config).label(frames)
    backward_raw = ExpensiveTeacher(config).label(list(reversed(frames)))
    backward = list(reversed(backward_raw))
    perturbed = [_lighting_perturb(frame, i) for i, frame in enumerate(frames)]
    light_labels = ExpensiveTeacher(config).label(perturbed)
    fb_error, fb_overlap = _paired_distance(forward, backward)
    light_error, light_overlap = _paired_distance(forward, light_labels)
    valid = [label for label in forward if label is not None]
    coverage = len(valid) / max(1, len(frames))
    confidence = mean(label.confidence for label in valid) if valid else 0.0
    radius = max(1, config.radius)
    fb_agreement = np.exp(-fb_error / radius) if np.isfinite(fb_error) else 0.0
    light_agreement = np.exp(-light_error / radius) if np.isfinite(light_error) else 0.0
    score = float(
        0.22 * min(1.0, coverage / 0.65)
        + 0.18 * confidence
        + 0.25 * fb_agreement * fb_overlap
        + 0.35 * light_agreement * light_overlap
    )
    return score, {
        "score": score,
        "coverage": coverage,
        "confidence": confidence,
        "forward_backward_error_px": fb_error,
        "forward_backward_overlap": fb_overlap,
        "lighting_error_px": light_error,
        "lighting_overlap": light_overlap,
    }


def tune_teacher(
    frames: list[np.ndarray],
    initial_center: tuple[int, int],
    initial_radius: int,
    trials: int = 18,
    seed: int = 7,
    progress=None,
) -> tuple[TeacherConfig, list[dict]]:
    sampled = sample_hsv_color(frames[0], initial_center, initial_radius)
    rng = np.random.default_rng(seed)
    candidates = [TeacherConfig(sampled, initial_radius)]
    for _ in range(max(0, trials - 1)):
        target = (
            int((sampled[0] + rng.integers(-4, 5)) % 180),
            int(np.clip(sampled[1] + rng.integers(-20, 21), 0, 255)),
            int(np.clip(sampled[2] + rng.integers(-20, 21), 0, 255)),
        )
        candidates.append(
            TeacherConfig(
                target_hsv=target,
                radius=max(3, initial_radius + int(rng.integers(-2, 3))),
                hue_tolerance=int(rng.integers(7, 19)),
                saturation_tolerance=int(rng.integers(45, 111)),
                value_tolerance=int(rng.integers(45, 121)),
                radius_tolerance=float(rng.uniform(0.18, 0.38)),
                learning_rate=float(rng.uniform(0.04, 0.16)),
            )
        )
    history = []
    best_config = candidates[0]
    best_score = -1.0
    for index, candidate in enumerate(candidates):
        if progress is not None:
            progress(
                "Teacher search",
                8 + 47 * index / max(1, len(candidates)),
                f"testing parameter set {index + 1}/{len(candidates)}",
            )
        score, metrics = teacher_self_supervision_score(frames, candidate)
        row = {"config": candidate.to_dict(), "metrics": metrics}
        history.append(row)
        if score > best_score:
            best_score = score
            best_config = candidate
    return best_config, history


def run_student_ablation(
    frames: list[np.ndarray],
    labels: list[PseudoLabel | None],
    teacher: TeacherConfig,
    target_latency_ms: float = 4.0,
    progress=None,
) -> list[AblationResult]:
    results = []
    combinations = [
        (preprocessing, detector, use_roi)
        for preprocessing in ("none", "open", "open_close", "gaussian")
        for detector in ("components", "matched_filter")
        for use_roi in (True, False)
    ]
    for combination_index, (preprocessing, detector, use_roi) in enumerate(combinations):
        if progress is not None:
            progress(
                "Student ablations",
                70 + 28 * combination_index / len(combinations),
                f"pipeline {combination_index + 1}/{len(combinations)}: {preprocessing}, {detector}, ROI={use_roi}",
            )
        config = FastTrackerConfig(
            radius=teacher.radius,
            radius_tolerance=teacher.radius_tolerance,
            hue_tolerance=teacher.hue_tolerance,
            saturation_tolerance=teacher.saturation_tolerance,
            value_tolerance=teacher.value_tolerance,
            learning_rate=teacher.learning_rate,
            preprocessing=preprocessing,
            detector=detector,
            use_roi=use_roi,
        )
        tracker = FastBallTracker(teacher.target_hsv, config)
        errors, latencies = [], []
        possible = 0
        matched = 0
        for frame, label in zip(frames, labels):
            result = tracker.detect_timed(frame)
            latencies.append(result.total_ms)
            if label is None:
                continue
            possible += 1
            if result.detection is not None:
                matched += 1
                errors.append(
                    float(
                        np.linalg.norm(
                            np.asarray(result.detection.center) - np.asarray(label.center)
                        )
                    )
                )
        recall = matched / max(1, possible)
        mean_error = mean(errors) if errors else float(teacher.radius * 4)
        p95_error = float(np.percentile(errors, 95)) if errors else float(teacher.radius * 4)
        mean_latency = mean(latencies)
        p95_latency = float(np.percentile(latencies, 95))
        objective = (
            mean_error / max(1, teacher.radius)
            + 1.5 * (1.0 - recall)
            + 0.20 * mean_latency / max(0.1, target_latency_ms)
        )
        results.append(
            AblationResult(
                preprocessing,
                detector,
                use_roi,
                mean_error,
                p95_error,
                recall,
                mean_latency,
                p95_latency,
                objective,
            )
        )
    return sorted(results, key=lambda item: item.objective)


def save_report(
    output_directory: str,
    teacher: TeacherConfig,
    teacher_search: list[dict],
    labels: list[PseudoLabel | None],
    ablations: list[AblationResult],
    elapsed_seconds: float,
    rectifier: FieldRectifier | None = None,
) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "teacher": teacher.to_dict(),
        "teacher_search": teacher_search,
        "pseudo_label_count": sum(label is not None for label in labels),
        "ablation_count": len(ablations),
        "recommended_student": asdict(ablations[0]),
        "elapsed_seconds": elapsed_seconds,
        "field_calibration": None
        if rectifier is None
        else {
            "corners": rectifier.points,
            "output_size": rectifier.output_size,
        },
        "warning": "Pseudo-label agreement is not independent ground truth; validate on a small manually labeled holdout before competition use.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "ablations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(ablations[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(item) for item in ablations)
    with (output / "pseudo_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["frame_index", "x", "y", "radius", "confidence", "votes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label in labels:
            if label is not None:
                writer.writerow(
                    {
                        "frame_index": label.frame_index,
                        "x": label.center[0],
                        "y": label.center[1],
                        "radius": label.radius,
                        "confidence": label.confidence,
                        "votes": label.votes,
                    }
                )


def tune_pipeline(
    video_path: str,
    initial_center: tuple[int, int],
    initial_radius: int,
    output_directory: str,
    maximum_frames: int = 600,
    teacher_trials: int = 18,
    rectifier: FieldRectifier | None = None,
    progress=None,
) -> tuple[TeacherConfig, list[AblationResult]]:
    started = perf_counter()
    if progress is not None:
        progress("Video preparation", 1, "loading and rectifying frames")
    frames = load_video_frames(video_path, maximum_frames, rectifier=rectifier)
    if progress is not None:
        progress("Video preparation", 7, f"prepared {len(frames)} frames")
    teacher_frames = frames[:: max(1, len(frames) // 90)]
    teacher, history = tune_teacher(
        teacher_frames,
        initial_center,
        initial_radius,
        teacher_trials,
        progress=progress,
    )
    if progress is not None:
        progress("Pseudo-labeling", 56, "running the selected ensemble teacher")
    teacher_model = ExpensiveTeacher(teacher)
    labels = []
    label_step = max(1, len(frames) // 12)
    for index, frame in enumerate(frames):
        labels.append(teacher_model.detect(frame, index))
        if progress is not None and index % label_step == 0:
            progress(
                "Pseudo-labeling",
                56 + 13 * index / max(1, len(frames)),
                f"frame {index + 1}/{len(frames)}",
            )
    ablations = run_student_ablation(
        frames, labels, teacher, progress=progress
    )
    if progress is not None:
        progress("Report", 99, "saving parameters, labels, and ablations")
    save_report(
        output_directory,
        teacher,
        history,
        labels,
        ablations,
        perf_counter() - started,
        rectifier,
    )
    if progress is not None:
        progress("Complete", 100, "tuning finished")
    return teacher, ablations
