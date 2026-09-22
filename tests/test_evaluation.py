import json

from robot_soccer_vision.evaluation import EvaluationAccumulator
from robot_soccer_vision.fast_tracker import TimedDetection
from robot_soccer_vision.teacher import PseudoLabel
from robot_soccer_vision.tracker import BallDetection


def timed(center=None, latency=1.0) -> TimedDetection:
    detection = None if center is None else BallDetection(center, 12.0, 0.9)
    return TimedDetection(detection, latency, latency * 0.6, latency * 0.3, True)


def label(center=None) -> PseudoLabel | None:
    if center is None:
        return None
    return PseudoLabel(0, center, 12.0, 0.85, 4)


def test_evaluation_counts_agreement_misses_and_fast_only_detections(tmp_path) -> None:
    evaluation = EvaluationAccumulator(ball_radius=12)
    error = evaluation.update(timed((10, 10)), label((13, 14)), 8.0, 2)
    evaluation.update(timed(None), label((20, 20)), 9.0)
    evaluation.update(timed((30, 30)), None, 7.0)

    assert error == 5.0
    summary = evaluation.summary()
    assert summary["processed_frames"] == 3
    assert summary["source_frames_skipped"] == 2
    assert summary["fast_recall_against_teacher"] == 0.5
    assert summary["fast_miss_rate_against_teacher"] == 0.5
    assert summary["fast_only_detection_rate"] == 1 / 3
    assert summary["center_error_px"]["mean"] == 5.0

    output = tmp_path / "evaluation.json"
    evaluation.save(str(output))
    assert json.loads(output.read_text())["matched_detections"] == 1


def test_empty_evaluation_has_serializable_null_distributions() -> None:
    summary = EvaluationAccumulator(ball_radius=12).summary()

    assert summary["center_error_px"]["mean"] is None
    assert summary["agreement_within_half_radius"] is None
