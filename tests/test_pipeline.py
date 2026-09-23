import json

import cv2
import numpy as np

from robot_soccer_vision.fast_tracker import FastBallTracker, FastTrackerConfig
from robot_soccer_vision.teacher import (
    ExpensiveTeacher,
    PseudoLabel,
    TeacherConfig,
    fuse_bidirectional_labels,
)
from robot_soccer_vision.tuning import run_student_ablation, save_report
from robot_soccer_vision.field_rectifier import FieldRectifier
from robot_soccer_vision.difficulty import rectifier_from_report


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
    assert result.refinement_ms > 0


def test_tracker_returns_to_full_field_after_roi_misses() -> None:
    tracker = FastBallTracker(
        (15, 220, 220),
        FastTrackerConfig(
            radius=14,
            recovery_scale=0.5,
            maximum_roi_misses=1,
            matched_filter_threshold=0.3,
        ),
    )
    assert tracker.detect_timed(scene((150, 100))).detection is not None
    blank = np.zeros((240, 320, 3), dtype=np.uint8)

    first_miss = tracker.detect_timed(blank)
    second_miss = tracker.detect_timed(blank)
    recovery = tracker.detect_timed(scene((260, 180)))

    assert first_miss.used_roi is True
    assert second_miss.used_roi is True
    assert recovery.used_roi is False
    assert recovery.detection is not None
    assert recovery.refinement_ms > 0


def test_selected_radius_precomputes_one_fixed_kernel() -> None:
    tracker = FastBallTracker((15, 220, 220), FastTrackerConfig(radius=11))

    assert tracker._kernel_radius == 11
    first_kernel = tracker._kernel
    tracker.set_radius(11)
    assert tracker._kernel is first_kernel
    tracker.set_radius(15)
    assert tracker._kernel_radius == 15
    assert tracker._kernel is not first_kernel


def test_expensive_teacher_reaches_consensus() -> None:
    config = TeacherConfig((15, 220, 220), radius=14)
    labels = ExpensiveTeacher(config).label(
        [scene((150 + offset * 3, 100 + offset), value=220 - offset * 10) for offset in range(4)]
    )

    assert all(label is not None for label in labels)
    assert all(label.votes >= 2 for label in labels if label is not None)
    assert np.linalg.norm(np.asarray(labels[-1].center) - (159, 103)) <= 3


def test_bidirectional_teacher_fuses_directions_and_fills_short_gap() -> None:
    def label(index, x, confidence=0.9):
        return PseudoLabel(index, (float(x), 100.0), 14.0, confidence, 4)

    forward = [label(0, 100), label(1, 104), None, label(3, 112), label(4, 116)]
    backward = [label(0, 101), label(1, 105), None, label(3, 113), label(4, 117)]

    fused, diagnostics = fuse_bidirectional_labels(forward, backward, 14, maximum_gap=2)

    assert all(item is not None for item in fused)
    assert diagnostics["paired_frames"] == 4
    assert diagnostics["interpolated_short_gap_frames"] == 1
    assert np.linalg.norm(np.asarray(fused[2].center) - (108.5, 100)) < 1.5


def test_bidirectional_teacher_rejects_direction_mismatch() -> None:
    forward = [PseudoLabel(0, (30.0, 40.0), 14.0, 0.9, 4)]
    backward = [PseudoLabel(0, (130.0, 140.0), 14.0, 0.9, 4)]

    fused, diagnostics = fuse_bidirectional_labels(forward, backward, 14)

    assert fused == [None]
    assert diagnostics["rejected_direction_disagreements"] == 1


def test_fast_tracker_result_cannot_depend_on_future_frame() -> None:
    config = FastTrackerConfig(radius=14, recovery_scale=1.0, matched_filter_threshold=0.3)
    first_tracker = FastBallTracker((15, 220, 220), config)
    second_tracker = FastBallTracker(
        (15, 220, 220),
        FastTrackerConfig(radius=14, recovery_scale=1.0, matched_filter_threshold=0.3),
    )
    prefix = [scene((140, 100)), scene((146, 103))]
    first_result = [first_tracker.detect_timed(frame).detection for frame in prefix][-1]
    second_result = [second_tracker.detect_timed(frame).detection for frame in prefix][-1]
    second_tracker.detect_timed(scene((280, 200)))

    assert first_result is not None and second_result is not None
    assert first_result.center == second_result.center


def test_ablation_and_report_are_serializable(tmp_path) -> None:
    frames = [scene((150 + i * 2, 100)) for i in range(3)]
    teacher = TeacherConfig((15, 220, 220), radius=14)
    labels = ExpensiveTeacher(teacher).label(frames)
    rectifier = FieldRectifier(minimum_dimension=20)
    for point in ((10, 10), (300, 15), (305, 220), (8, 225)):
        rectifier.add_point(point)

    results = run_student_ablation(frames, labels, teacher)
    temporal = {
        "mode": "bidirectional_rts",
        "paired_frames": 3,
        "interpolated_short_gap_frames": 0,
    }
    save_report(
        str(tmp_path),
        teacher,
        [],
        labels,
        results,
        1.0,
        rectifier,
        temporal_labeling=temporal,
    )

    report = json.loads((tmp_path / "report.json").read_text())
    assert len(results) == 16
    assert report["recommended_student"]["objective"] == results[0].objective
    assert report["temporal_labeling"] == temporal
    restored = rectifier_from_report(report)
    assert restored is not None and restored.ready
    assert restored.output_size == rectifier.output_size
    assert (tmp_path / "ablations.csv").exists()
    assert (tmp_path / "pseudo_labels.csv").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["run_type"] == "self_supervised_tuning"
    assert {item["filename"] for item in manifest["artifacts"]} == {
        "report.json",
        "ablations.csv",
        "pseudo_labels.csv",
    }
