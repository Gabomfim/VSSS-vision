"""Production-oriented low-latency adaptive camera application."""

import argparse
import json
from pathlib import Path
from time import monotonic

import cv2

from .app import _draw_label, _parse_source
from .fast_tracker import FastBallTracker, FastTrackerConfig
from .latest_frame import LatestFrameCapture
from .tracker import sample_hsv_color

WINDOW = "Robot Soccer - Fast Adaptive Tracker"
CONTROLS = "Fast calibration"


def _configuration_from_report(path: str | None) -> tuple[FastTrackerConfig, tuple[int, int, int]]:
    if path is None:
        return FastTrackerConfig(), (15, 220, 220)
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    teacher = report["teacher"]
    student = report["recommended_student"]
    config = FastTrackerConfig(
        radius=int(teacher["radius"]),
        radius_tolerance=float(teacher["radius_tolerance"]),
        hue_tolerance=int(teacher["hue_tolerance"]),
        saturation_tolerance=int(teacher["saturation_tolerance"]),
        value_tolerance=int(teacher["value_tolerance"]),
        learning_rate=float(teacher["learning_rate"]),
        preprocessing=student["preprocessing"],
        detector=student["detector"],
        use_roi=bool(student["use_roi"]),
    )
    return config, tuple(teacher["target_hsv"])


def run(source: int | str = 0, report_path: str | None = None) -> None:
    config, target = _configuration_from_report(report_path)
    capture = LatestFrameCapture(source)
    tracker = FastBallTracker(target, config)
    cursor = [0, 0]
    calibrating = report_path is None
    picked = report_path is not None
    sequence = 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(CONTROLS, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Ball radius", CONTROLS, config.radius, 100, lambda _v: None)
    cv2.createTrackbar("Learning rate %", CONTROLS, round(config.learning_rate * 100), 50, lambda _v: None)

    def mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        nonlocal calibrating
        cursor[:] = [x, y]
        if event == cv2.EVENT_LBUTTONDOWN:
            calibrating = True

    cv2.setMouseCallback(WINDOW, mouse)
    try:
        while True:
            ok, frame, sequence, captured_at = capture.read(sequence)
            if not ok:
                if capture.finished:
                    break
                continue
            if cursor == [0, 0]:
                cursor[:] = [frame.shape[1] // 2, frame.shape[0] // 2]
            config.radius = max(2, cv2.getTrackbarPos("Ball radius", CONTROLS))
            config.learning_rate = max(0.01, cv2.getTrackbarPos("Learning rate %", CONTROLS) / 100)
            display = frame.copy()
            if calibrating:
                cv2.circle(display, tuple(cursor), config.radius, (255, 255, 255), 2)
                _draw_label(display, "FAST CALIBRATION: place circle over ball", 0, (0, 255, 255))
                _draw_label(display, "Press SPACE to sample and track", 1)
            elif picked:
                result = tracker.detect_timed(frame)
                age_ms = (monotonic() - captured_at) * 1000
                if result.detection is not None:
                    detection = result.detection
                    cv2.circle(display, detection.center, round(detection.radius), (0, 255, 0), 2)
                    cv2.drawMarker(display, detection.center, (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
                    _draw_label(display, f"BALL x={detection.center[0]} y={detection.center[1]}", 0, (0, 255, 0))
                else:
                    _draw_label(display, "Ball not found - recovering globally", 0, (0, 0, 255))
                _draw_label(
                    display,
                    f"processing={result.total_ms:.2f}ms frame-age={age_ms:.1f}ms ROI={result.used_roi}",
                    1,
                )
                _draw_label(display, "C/click: recalibrate   A: freeze/resume learning", 2)
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                calibrating = True
            if key == ord("a"):
                tracker.adaptation_enabled = not tracker.adaptation_enabled
            if key == ord(" ") and calibrating:
                tracker.target_hsv = sample_hsv_color(frame, tuple(cursor), config.radius)
                tracker.reset_motion()
                tracker.update_count = 0
                picked = True
                calibrating = False
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tuned low-latency adaptive tracker")
    parser.add_argument("--source", default="0", help="Camera index or path to video")
    parser.add_argument("--report", help="report.json produced by robot-soccer-tune")
    args = parser.parse_args()
    run(_parse_source(args.source), args.report)


if __name__ == "__main__":
    main()
