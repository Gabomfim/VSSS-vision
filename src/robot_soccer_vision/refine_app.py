"""Refine a self-supervised calibration with actively selected manual labels."""

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import cv2
import numpy as np

from .difficulty import rectifier_from_report
from .provenance import file_record, runtime_record
from .teacher import ExpensiveTeacher, TeacherConfig
from .tracker import sample_hsv_color
from .tuning import load_video_frames, run_student_ablation, save_report


def _config(item: dict) -> TeacherConfig:
    return TeacherConfig(
        tuple(item["target_hsv"]), int(item["radius"]),
        int(item["hue_tolerance"]), int(item["saturation_tolerance"]),
        int(item["value_tolerance"]), float(item["radius_tolerance"]),
        float(item["learning_rate"]), int(item.get("minimum_votes", 2)),
    )


def refine(source: str, report_path: str, labels_path: str, output: str, maximum_frames: int) -> None:
    started = perf_counter()
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    rectifier = rectifier_from_report(report)
    with Path(labels_path).open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] in ("labeled", "absent")]
    positives = [row for row in rows if row["status"] == "labeled"]
    negatives = [row for row in rows if row["status"] == "absent"]
    if len(positives) < 3:
        raise ValueError("At least three manually labeled frames are required")

    wanted = {int(row["frame_index"]): row for row in rows}
    capture = cv2.VideoCapture(source)
    labeled_frames: dict[int, np.ndarray] = {}
    index = 0
    try:
        while wanted.keys() - labeled_frames.keys():
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                labeled_frames[index] = rectifier.warp(frame) if rectifier else frame
            index += 1
    finally:
        capture.release()
    if len(labeled_frames) < 3:
        raise ValueError("Could not load at least three labeled frames from the source video")

    candidates = [_config(report["teacher"])]
    candidates.extend(_config(item["config"]) for item in report.get("teacher_search", []))
    samples = []
    radii = []
    for frame_index, frame in labeled_frames.items():
        row = wanted[frame_index]
        if row["status"] != "labeled":
            continue
        center = (round(float(row["x"])), round(float(row["y"])))
        radius = max(2, round(float(row["radius"])))
        samples.append(sample_hsv_color(frame, center, radius))
        radii.append(radius)
    base = candidates[0]
    hsv = tuple(int(median(channel)) for channel in zip(*samples))
    candidates.append(replace(base, target_hsv=hsv, radius=round(median(radii))))

    scored = []
    for candidate in candidates:
        detector = ExpensiveTeacher(candidate)
        errors = []
        found = 0
        false_positives = 0
        for frame_index in sorted(labeled_frames):
            label = detector.detect(labeled_frames[frame_index], frame_index)
            row = wanted[frame_index]
            if row["status"] == "absent":
                false_positives += label is not None
                continue
            if label is None:
                continue
            found += 1
            errors.append(float(np.linalg.norm(np.asarray(label.center) - (float(row["x"]), float(row["y"])))))
        recall = found / len(positives)
        false_positive_rate = false_positives / max(1, len(negatives))
        mean_error = mean(errors) if errors else candidate.radius * 4.0
        objective = mean_error / max(1, candidate.radius) + 1.5 * (1.0 - recall) + false_positive_rate
        scored.append({"config": candidate.to_dict(), "manual_mean_error_px": mean_error, "manual_recall": recall, "manual_false_positive_rate": false_positive_rate, "manual_objective": objective})
    scored.sort(key=lambda item: item["manual_objective"])
    teacher = _config(scored[0]["config"])

    print(f"Selected active-learning teacher from {len(rows)} labels; loading training frames...")
    frames = load_video_frames(source, maximum_frames, rectifier=rectifier)
    model = ExpensiveTeacher(teacher)
    pseudo_labels = model.label_bidirectional(frames)
    ablations = run_student_ablation(frames, pseudo_labels, teacher)
    provenance = {
        "inputs": [file_record(source, "calibration_video"), file_record(report_path, "base_calibration_report"), file_record(labels_path, "active_learning_labels")],
        "runtime": runtime_record(Path(__file__).resolve().parents[2]),
    }
    active_learning = {
        "labeled_frames": len(rows),
        "positive_frames": len(positives),
        "negative_frames": len(negatives),
        "candidate_scores": scored,
        "selected_manual_mean_error_px": scored[0]["manual_mean_error_px"],
        "selected_manual_recall": scored[0]["manual_recall"],
    }
    save_report(output, teacher, report.get("teacher_search", []), pseudo_labels, ablations,
                perf_counter() - started, rectifier, provenance,
                model.bidirectional_diagnostics, active_learning)
    print(f"Active-learning calibration written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine calibration using active-learning manual labels")
    parser.add_argument("--source", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=600)
    args = parser.parse_args()
    refine(args.source, args.report, args.labels, args.output, args.max_frames)


if __name__ == "__main__":
    main()
