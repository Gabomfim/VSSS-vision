"""Render every source frame with the calibrated fast tracker for demonstration."""

import argparse
import json
from pathlib import Path

import cv2

from robot_soccer_vision.app import _draw_label
from robot_soccer_vision.difficulty import rectifier_from_report
from robot_soccer_vision.fast_app import _configuration_from_report
from robot_soccer_vision.fast_tracker import FastBallTracker


def render(source: str, report_path: str, output: str) -> None:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    rectifier = rectifier_from_report(report)
    if rectifier is None or not rectifier.ready:
        raise ValueError("The report does not contain a valid field calibration")

    config, target = _configuration_from_report(report_path)
    tracker = FastBallTracker(target, config)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
    total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width, height = rectifier.output_size
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create demonstration video: {destination}")

    frame_index = 0
    try:
        while True:
            ok, raw = capture.read()
            if not ok:
                break
            frame = rectifier.warp(raw)
            result = tracker.detect_timed(frame)
            display = frame.copy()
            if result.detection is None:
                _draw_label(display, "Ball not found - recovering globally", 0, (0, 0, 255))
            else:
                detection = result.detection
                cv2.circle(display, detection.center, round(detection.radius), (0, 255, 0), 2)
                cv2.drawMarker(
                    display, detection.center, (0, 255, 0), cv2.MARKER_CROSS, 16, 2
                )
                _draw_label(
                    display,
                    f"BALL x={detection.center[0]} y={detection.center[1]}",
                    0,
                    (0, 255, 0),
                )
            _draw_label(display, f"FAST tracker frame={frame_index}", 1)
            writer.write(display)
            frame_index += 1
            if frame_index % 300 == 0 or frame_index == total:
                percent = 100.0 if total == 0 else frame_index * 100.0 / total
                print(f"Rendered {frame_index}/{total or '?'} frames ({percent:.1f}%)", flush=True)
    finally:
        capture.release()
        writer.release()

    print(f"Demonstration written to {destination} with {frame_index} frames", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(args.source, args.report, args.output)


if __name__ == "__main__":
    main()
