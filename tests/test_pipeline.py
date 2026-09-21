import json

import cv2
import numpy as np

from robot_soccer_vision.fast_tracker import FastBallTracker, FastTrackerConfig
from robot_soccer_vision.teacher import ExpensiveTeacher, TeacherConfig
from robot_soccer_vision.tuning import run_student_ablation, save_report


def scene(center=(160, 110), radius=14, value=220) -> np.ndarray:
    hsv = np.zeros((240, 320, 3), dtype=np.uint8)
    hsv[:, :] = (60, 100, 70)
    cv2.circle(hsv, center, radius, (15, 220, value), -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_fast_matched_filter_uses_roi_after_first_detection() -> None:
    config = FastTrackerConfig(
        radius=14,
        recovery_scale=1.0,
        detector="matched_filter",
        matched_filter_threshold=0.3,
    )
    tracker = FastBallTracker((15, 220, 220), config)

    first = tracker.detect_timed(scene((140, 100)))
    second = tracker.detect_timed(scene((146, 103)))

    assert first.detection is not None
    assert second.detection is not None
    assert first.used_roi is False
    assert second.used_roi is True
    assert np.linalg.norm(np.asarray(second.detection.center) - (146, 103)) <= 2


def test_downsampled_recovery_maps_to_full_resolution() -> None:
    tracker = FastBallTracker(
        (15, 220, 220),
        FastTrackerConfig(radius=14, recovery_scale=0.5, matched_filter_threshold=0.25),
    )

    result = tracker.detect_timed(scene((200, 120)))

    assert result.detection is not None
    assert np.linalg.norm(np.asarray(result.detection.center) - (200, 120)) <= 3


def test_expensive_teacher_reaches_consensus() -> None:
    config = TeacherConfig((15, 220, 220), radius=14)
    labels = ExpensiveTeacher(config).label(
        [scene((150 + offset * 3, 100 + offset), value=220 - offset * 10) for offset in range(4)]
    )

    assert all(label is not None for label in labels)
    assert all(label.votes >= 2 for label in labels if label is not None)
    assert np.linalg.norm(np.asarray(labels[-1].center) - (159, 103)) <= 3


def test_ablation_and_report_are_serializable(tmp_path) -> None:
    frames = [scene((150 + i * 2, 100)) for i in range(3)]
    teacher = TeacherConfig((15, 220, 220), radius=14)
    labels = ExpensiveTeacher(teacher).label(frames)

    results = run_student_ablation(frames, labels, teacher)
    save_report(str(tmp_path), teacher, [], labels, results, 1.0)

    report = json.loads((tmp_path / "report.json").read_text())
    assert len(results) == 16
    assert report["recommended_student"]["objective"] == results[0].objective
    assert (tmp_path / "ablations.csv").exists()
    assert (tmp_path / "pseudo_labels.csv").exists()
