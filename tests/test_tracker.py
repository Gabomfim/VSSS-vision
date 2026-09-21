import cv2
import numpy as np

from robot_soccer_vision.tracker import BallTracker, TrackerConfig, sample_hsv_color
from robot_soccer_vision.adaptive_tracker import AdaptiveBallTracker, AdaptiveTrackerConfig


ORANGE = (0, 140, 255)  # BGR


def test_tracks_orange_circle_at_expected_size() -> None:
    frame = np.full((240, 320, 3), (30, 90, 30), dtype=np.uint8)
    cv2.circle(frame, (190, 110), 14, ORANGE, -1)
    target = sample_hsv_color(frame, (190, 110), 14)
    tracker = BallTracker(target, TrackerConfig(radius=14, radius_tolerance=0.25))

    detection = tracker.detect(frame)

    assert detection is not None
    assert abs(detection.center[0] - 190) <= 1
    assert abs(detection.center[1] - 110) <= 1
    assert detection.score > 0.8


def test_rejects_correct_color_with_wrong_size() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(frame, (100, 100), 35, ORANGE, -1)
    target = sample_hsv_color(frame, (100, 100), 20)
    tracker = BallTracker(target, TrackerConfig(radius=12, radius_tolerance=0.2))

    assert tracker.detect(frame) is None


def test_rejects_expected_circle_with_wrong_color() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(frame, (100, 100), 12, (255, 0, 0), -1)
    orange_hsv = tuple(int(v) for v in cv2.cvtColor(np.uint8([[ORANGE]]), cv2.COLOR_BGR2HSV)[0, 0])
    tracker = BallTracker(orange_hsv, TrackerConfig(radius=12))

    assert tracker.detect(frame) is None


def test_hue_threshold_wraps_at_red_boundary() -> None:
    hsv = np.zeros((80, 80, 3), dtype=np.uint8)
    hsv[:] = (179, 240, 240)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    tracker = BallTracker((1, 240, 240), TrackerConfig(radius=12, hue_tolerance=3))

    assert np.all(tracker.color_mask(frame) == 255)


def _hsv_ball(value: int, hue: int = 15) -> np.ndarray:
    hsv = np.zeros((180, 240, 3), dtype=np.uint8)
    hsv[:, :] = (60, 100, 70)
    cv2.circle(hsv, (120, 90), 14, (hue, 220, value), -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_adaptive_tracker_follows_gradual_light_change() -> None:
    config = AdaptiveTrackerConfig(
        radius=14,
        hue_tolerance=8,
        saturation_tolerance=35,
        value_tolerance=28,
        learning_rate=0.5,
    )
    tracker = AdaptiveBallTracker((15, 220, 220), config)

    detections = [tracker.detect_and_update(_hsv_ball(value)) for value in range(210, 109, -10)]

    assert all(detection is not None for detection in detections)
    assert tracker.target_hsv[2] < 135
    assert tracker.update_count == len(detections)


def test_adaptive_tracker_does_not_learn_without_a_circle() -> None:
    tracker = AdaptiveBallTracker((15, 220, 220), AdaptiveTrackerConfig(radius=14))
    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    frame[:, 80:160] = ORANGE

    assert tracker.detect_and_update(frame) is None
    assert tracker.target_hsv == (15, 220, 220)
    assert tracker.update_count == 0
