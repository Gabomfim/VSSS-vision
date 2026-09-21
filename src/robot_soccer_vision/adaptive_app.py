"""Interactive ball tracker with online illumination adaptation."""

import argparse
import time

import cv2

from .adaptive_tracker import AdaptiveBallTracker, AdaptiveTrackerConfig
from .app import _draw_label, _parse_source
from .tracker import sample_hsv_color

WINDOW = "Robot Soccer - Adaptive Ball Tracker"
CONTROLS = "Adaptive calibration"


def run(source: int | str = 0) -> None:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera or video source: {source}")

    config = AdaptiveTrackerConfig()
    tracker = AdaptiveBallTracker(config=config)
    cursor = [0, 0]
    calibrating = True
    picked = False
    last_detection = None
    last_seen = 0.0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(CONTROLS, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Ball radius", CONTROLS, config.radius, 100, lambda _v: None)
    cv2.createTrackbar("Hue tolerance", CONTROLS, config.hue_tolerance, 45, lambda _v: None)
    cv2.createTrackbar("Saturation tolerance", CONTROLS, config.saturation_tolerance, 255, lambda _v: None)
    cv2.createTrackbar("Brightness tolerance", CONTROLS, config.value_tolerance, 255, lambda _v: None)
    cv2.createTrackbar("Learning rate %", CONTROLS, round(config.learning_rate * 100), 50, lambda _v: None)

    def mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        nonlocal calibrating
        cursor[:] = [x, y]
        if event == cv2.EVENT_LBUTTONDOWN:
            calibrating = True

    cv2.setMouseCallback(WINDOW, mouse)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if cursor == [0, 0]:
                cursor[:] = [frame.shape[1] // 2, frame.shape[0] // 2]

            config.radius = max(2, cv2.getTrackbarPos("Ball radius", CONTROLS))
            config.hue_tolerance = max(1, cv2.getTrackbarPos("Hue tolerance", CONTROLS))
            config.saturation_tolerance = cv2.getTrackbarPos("Saturation tolerance", CONTROLS)
            config.value_tolerance = cv2.getTrackbarPos("Brightness tolerance", CONTROLS)
            config.learning_rate = max(0.01, cv2.getTrackbarPos("Learning rate %", CONTROLS) / 100.0)

            display = frame.copy()
            if calibrating:
                cv2.circle(display, tuple(cursor), config.radius, (255, 255, 255), 2)
                cv2.drawMarker(display, tuple(cursor), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
                _draw_label(display, "ADAPTIVE CALIBRATION: place circle over ball", 0, (0, 255, 255))
                _draw_label(display, "Adjust radius, then press SPACE to pick color", 1)
            elif picked:
                detection = tracker.detect_and_update(frame)
                if detection is not None:
                    last_detection = detection
                    last_seen = time.monotonic()
                if last_detection is not None and time.monotonic() - last_seen < 0.2:
                    center = last_detection.center
                    cv2.circle(display, center, round(last_detection.radius), (0, 255, 0), 2)
                    cv2.drawMarker(display, center, (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
                    _draw_label(display, f"BALL x={center[0]} y={center[1]} score={last_detection.score:.2f}", 0, (0, 255, 0))
                else:
                    _draw_label(display, "Ball not found - color model held", 0, (0, 0, 255))
                state = "LEARNING" if tracker.adaptation_enabled else "FROZEN"
                h, s, v = tracker.target_hsv
                _draw_label(display, f"{state}  HSV=({h}, {s}, {v})  updates={tracker.update_count}", 1, (0, 255, 255))
                _draw_label(display, "A: freeze/resume learning   C/click: recalibrate", 2)

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
                tracker.update_count = 0
                picked = True
                calibrating = False
                last_detection = None
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track a colored ball while adapting to illumination changes"
    )
    parser.add_argument("--source", default="0", help="Camera index or path to a video file")
    args = parser.parse_args()
    run(_parse_source(args.source))


if __name__ == "__main__":
    main()

