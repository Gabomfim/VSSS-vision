"""Manual review UI for randomly sampled difficult frames."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone

import cv2

from .difficulty import (
    configurations_from_report,
    load_selected_frames,
    rectifier_from_report,
    sample_difficult_frames,
    score_video_difficulty,
)
from .provenance import artifact_records, file_record, runtime_record, write_manifest

WINDOW = "Robot Soccer - Difficult Frame Labeler"
FIELDNAMES = [
    "frame_index",
    "status",
    "skip_reason",
    "x",
    "y",
    "radius",
    "difficulty",
    "teacher_x",
    "teacher_y",
    "teacher_radius",
    "teacher_confidence",
    "teacher_votes",
    "student_teacher_distance",
    "temporal_surprise",
    "blur_score",
    "median_brightness",
    "shadow_fraction",
]


def _existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save_label_manifest(
    destination: Path,
    selection_path: Path,
    provenance: dict,
    rows: list[dict],
) -> None:
    artifacts = [selection_path]
    if destination.exists():
        artifacts.insert(0, destination)
    write_manifest(
        destination.with_name(destination.stem + ".manifest.json"),
        {
            "schema_version": 1,
            "run_type": "manual_difficult_frame_labeling",
            "provenance": provenance,
            "counts": {
                "total_reviewed": len(rows),
                "labeled": sum(row["status"] == "labeled" for row in rows),
                "absent": sum(row["status"] == "absent" for row in rows),
                "skipped": sum(row["status"] == "skipped" for row in rows),
            },
            "artifacts": artifact_records(artifacts),
        },
    )


def _annotation_row(record, status: str, center, radius: int, reason: str = "") -> dict:
    row = asdict(record)
    row.update(
        {
            "status": status,
            "skip_reason": reason,
            "x": "" if center is None else center[0],
            "y": "" if center is None else center[1],
            "radius": "" if center is None else radius,
        }
    )
    return {field: row.get(field, "") for field in FIELDNAMES}


def annotate(video_path: str, report_path: str, output_path: str, count: int, pool_fraction: float, seed: int, maximum_frames: int | None, stride: int) -> None:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    teacher, student = configurations_from_report(report)
    rectifier = rectifier_from_report(report)
    destination = Path(output_path)
    rows = _existing_rows(destination)
    completed = {int(row["frame_index"]) for row in rows}
    print("Recording input provenance and SHA-256 hashes...")
    provenance = {
        "inputs": [
            file_record(video_path, "manual_validation_video"),
            file_record(report_path, "calibration_report"),
        ],
        "calibration_provenance": report.get("provenance"),
        "sampling": {
            "requested_count": count,
            "difficult_pool_fraction": pool_fraction,
            "random_seed": seed,
            "maximum_frames": maximum_frames,
            "stride": stride,
        },
        "runtime": runtime_record(Path(__file__).resolve().parents[2]),
    }
    print("Scoring video difficulty; the expensive teacher may take a while...")
    records = score_video_difficulty(
        video_path,
        teacher,
        student,
        maximum_frames=maximum_frames,
        stride=stride,
        rectifier=rectifier,
    )
    selected = sample_difficult_frames(records, count, pool_fraction, seed, completed)
    frames = load_selected_frames(video_path, selected, rectifier)
    selection_path = destination.with_name(destination.stem + ".selection.json")
    selection_history = []
    if selection_path.exists():
        try:
            selection_history = json.loads(selection_path.read_text(encoding="utf-8"))[
                "runs"
            ]
        except (KeyError, json.JSONDecodeError):
            selection_history = []
    selection_history.append(
        {
            "selected_utc": datetime.now(timezone.utc).isoformat(),
            "sampling": provenance["sampling"],
            "frames": [item.to_dict() for item in selected],
        }
    )
    write_manifest(
        selection_path,
        {"schema_version": 1, "runs": selection_history},
    )
    _save_label_manifest(destination, selection_path, provenance, rows)
    if not selected:
        print("No unlabeled frames were available.")
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    state = {"center": None, "radius": teacher.radius}

    def mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["center"] = (x, y)

    cv2.setMouseCallback(WINDOW, mouse)
    position = 0
    try:
        while position < len(selected):
            record = selected[position]
            frame = frames.get(record.frame_index)
            if frame is None:
                rows.append(_annotation_row(record, "skipped", None, state["radius"], "unreadable"))
                _save_rows(destination, rows)
                _save_label_manifest(destination, selection_path, provenance, rows)
                position += 1
                continue
            if state["center"] is None and record.teacher_x is not None:
                state["center"] = (round(record.teacher_x), round(record.teacher_y))
                state["radius"] = round(record.teacher_radius or teacher.radius)
            display = frame.copy()
            if state["center"] is not None:
                cv2.circle(display, state["center"], state["radius"], (0, 255, 255), 2)
                cv2.drawMarker(display, state["center"], (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
            cv2.putText(display, f"{position + 1}/{len(selected)} frame={record.frame_index} difficulty={record.difficulty:.2f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            cv2.putText(display, f"confidence={record.teacher_confidence:.2f} votes={record.teacher_votes} blur={record.blur_score:.0f} brightness={record.median_brightness:.0f}", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(display, "Click center; +/- radius; ENTER save", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(display, "N: ball absent   Skip: M blur  D dark  O occluded  S other", (12, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("+"), ord("=")):
                state["radius"] += 1
                continue
            if key in (ord("-"), ord("_")):
                state["radius"] = max(2, state["radius"] - 1)
                continue
            skip_reasons = {
                ord("m"): "motion_blur",
                ord("M"): "motion_blur",
                ord("d"): "darkness",
                ord("D"): "darkness",
                ord("o"): "occlusion",
                ord("O"): "occlusion",
                ord("s"): "other_impossible",
                ord("S"): "other_impossible",
            }
            if key in (ord("n"), ord("N")):
                rows.append(_annotation_row(record, "absent", None, state["radius"]))
            elif key in skip_reasons:
                rows.append(_annotation_row(record, "skipped", None, state["radius"], skip_reasons[key]))
            elif key in (10, 13) and state["center"] is not None:
                rows.append(_annotation_row(record, "labeled", state["center"], state["radius"]))
            else:
                continue
            _save_rows(destination, rows)
            _save_label_manifest(destination, selection_path, provenance, rows)
            position += 1
            state["center"] = None
    finally:
        cv2.destroyAllWindows()
    print(f"Saved {len(rows)} total annotations to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample and manually label difficult ball frames")
    parser.add_argument("--source", required=True, help="Recorded video")
    parser.add_argument("--report", required=True, help="Tuning report.json")
    parser.add_argument("--count", type=int, default=100, help="Number of difficult frames to sample")
    parser.add_argument("--pool-fraction", type=float, default=0.25, help="Top difficult fraction forming the random pool")
    parser.add_argument("--seed", type=int, default=7, help="Reproducible random seed")
    parser.add_argument("--output", default="manual-labels.csv")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    annotate(args.source, args.report, args.output, args.count, args.pool_fraction, args.seed, args.max_frames, args.stride)


if __name__ == "__main__":
    main()
