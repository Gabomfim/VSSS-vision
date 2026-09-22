"""Production-oriented low-latency adaptive camera application."""

import argparse
import json
from pathlib import Path
from time import monotonic, perf_counter_ns

import cv2

from .app import _draw_label, _parse_source
from .fast_tracker import FastBallTracker, FastTrackerConfig
from .latest_frame import LatestFrameCapture
from .tracker import sample_hsv_color
from .field_rectifier import FieldRectifier
from .difficulty import configurations_from_report
from .evaluation import EvaluationAccumulator
from .teacher import ExpensiveTeacher

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


def run(
    source: int | str = 0,
    report_path: str | None = None,
    evaluate: bool = False,
    evaluation_output: str = "fast-evaluation.json",
) -> None:
    if evaluate and report_path is None:
        raise ValueError("--evaluate requires --report so the robust teacher is defined")
    config, target = _configuration_from_report(report_path)
    capture = LatestFrameCapture(source, start_paused_after_first=True)
    tracker = FastBallTracker(target, config)
    cursor = [0, 0]
    calibrating = report_path is None
    picked = report_path is not None
    sequence = 0
    rectifier = FieldRectifier()
    field_error = [""]
    raw_frame = None
    captured_at = 0.0
    teacher = None
    evaluation = None
    evaluation_frame = 0
    previous_source_sequence = 0
    latest_error = None
    if evaluate:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        teacher_config, _ = configurations_from_report(report)
        teacher = ExpensiveTeacher(teacher_config)
        evaluation = EvaluationAccumulator(teacher_config.radius)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(CONTROLS, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Ball radius", CONTROLS, config.radius, 100, lambda _v: None)
    cv2.createTrackbar("Learning rate %", CONTROLS, round(config.learning_rate * 100), 50, lambda _v: None)

    def mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        nonlocal calibrating
        if event == cv2.EVENT_LBUTTONDOWN:
            if not rectifier.ready:
                try:
                    rectifier.add_point((x, y))
                    field_error[0] = ""
                    if rectifier.ready and picked and not calibrating:
                        capture.resume()
                except ValueError as error:
                    field_error[0] = str(error)
            else:
                cursor[:] = [x, y]
                calibrating = True
                capture.pause()

    cv2.setMouseCallback(WINDOW, mouse)
    try:
        while True:
            needs_frame = raw_frame is None or (
                rectifier.ready and picked and not calibrating
            )
            if needs_frame:
                ok, new_frame, sequence, captured_at = capture.read(sequence)
                if not ok:
                    if capture.finished:
                        break
                    continue
                raw_frame = new_frame
                skipped_source_frames = max(0, sequence - previous_source_sequence - 1)
                previous_source_sequence = sequence
            else:
                skipped_source_frames = 0
            frame = raw_frame
            if not rectifier.ready:
                display = rectifier.draw_setup(frame)
                next_name = rectifier.next_corner_name or "correct the selection"
                _draw_label(display, f"FIELD SETUP: click {next_name}", 0, (0, 255, 255))
                _draw_label(display, "Order: top-left, top-right, bottom-right, bottom-left", 1)
                _draw_label(display, "U: undo   R: restart   Q: quit", 2)
                if field_error[0]:
                    _draw_label(display, field_error[0], 3, (0, 0, 255))
                cv2.imshow(WINDOW, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("u"):
                    rectifier.undo()
                    field_error[0] = ""
                if key == ord("r"):
                    rectifier.reset()
                    field_error[0] = ""
                continue
            warp_started = perf_counter_ns()
            frame = rectifier.warp(frame)
            warp_ms = (perf_counter_ns() - warp_started) / 1e6
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
                if evaluation is not None and teacher is not None:
                    teacher_started = perf_counter_ns()
                    teacher_label = teacher.detect(frame, evaluation_frame)
                    teacher_ms = (perf_counter_ns() - teacher_started) / 1e6
                    latest_error = evaluation.update(
                        result,
                        teacher_label,
                        teacher_ms,
                        skipped_source_frames,
                    )
                    evaluation_frame += 1
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
                    f"total={result.total_ms + warp_ms:.2f}ms warp={warp_ms:.2f}ms frame-age={age_ms:.1f}ms ROI={result.used_roi}",
                    1,
                )
                _draw_label(display, "C: ball   F: field corners   A: freeze learning", 2)
                if evaluation is not None:
                    error_text = "n/a" if latest_error is None else f"{latest_error:.2f}px"
                    _draw_label(
                        display,
                        f"EVAL teacher difference={error_text} matched={evaluation.matched_detections}/{evaluation.teacher_detections}",
                        3,
                        (255, 200, 0),
                    )
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                calibrating = True
                capture.pause()
            if key == ord("f"):
                capture.pause()
                rectifier.reset()
                calibrating = True
                picked = False
                cursor[:] = [0, 0]
            if key == ord("a"):
                tracker.adaptation_enabled = not tracker.adaptation_enabled
            if key == ord(" ") and calibrating:
                tracker.target_hsv = sample_hsv_color(frame, tuple(cursor), config.radius)
                tracker.reset_motion()
                tracker.update_count = 0
                picked = True
                calibrating = False
                capture.resume()
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if evaluation is not None:
            evaluation.save(evaluation_output)
            print(f"Evaluation saved to {evaluation_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tuned low-latency adaptive tracker")
    parser.add_argument("--source", default="0", help="Camera index or path to video")
    parser.add_argument("--report", help="report.json produced by robot-soccer-tune")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="compare the fast tracker with the report's expensive adaptive teacher",
    )
    parser.add_argument(
        "--evaluation-output",
        default="fast-evaluation.json",
        help="JSON destination used with --evaluate",
    )
    args = parser.parse_args()
    if args.evaluate and not args.report:
        parser.error("--evaluate requires --report")
    run(_parse_source(args.source), args.report, args.evaluate, args.evaluation_output)


if __name__ == "__main__":
    main()
