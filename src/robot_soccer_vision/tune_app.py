"""Command-line entry point for self-supervised tuning and ablations."""

import argparse

from .tuning import tune_pipeline
from .tuning_calibration import calibrate_tuning_frame, first_video_frame


class ConsoleProgress:
    def __init__(self) -> None:
        self.last_percent = -1
        self.last_stage = ""

    def __call__(self, stage: str, percent: float, message: str) -> None:
        whole = max(0, min(100, round(percent)))
        if whole == self.last_percent and stage == self.last_stage:
            return
        self.last_percent = whole
        self.last_stage = stage
        filled = whole // 4
        bar = "#" * filled + "-" * (25 - filled)
        print(f"\r[{bar}] {whole:3d}%  {stage}: {message}", end="", flush=True)
        if whole == 100:
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune the expensive teacher, generate pseudo-labels, and benchmark fast trackers"
    )
    parser.add_argument("--source", required=True, help="Recorded calibration/match video")
    parser.add_argument("--output", default="tuning-results", help="Report output directory")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--teacher-trials", type=int, default=18)
    args = parser.parse_args()

    first_frame = first_video_frame(args.source)
    rectifier, initial_center, initial_radius = calibrate_tuning_frame(first_frame)
    progress = ConsoleProgress()

    teacher, ablations = tune_pipeline(
        args.source,
        initial_center,
        initial_radius,
        args.output,
        args.max_frames,
        args.teacher_trials,
        rectifier=rectifier,
        progress=progress,
    )
    best = ablations[0]
    print(f"Best teacher HSV: {teacher.target_hsv}; radius: {teacher.radius}")
    print(
        "Recommended student: "
        f"preprocessing={best.preprocessing}, detector={best.detector}, ROI={best.use_roi}"
    )
    print(
        f"Mean error={best.mean_center_error_px:.2f}px, recall={best.recall:.1%}, "
        f"mean processing={best.mean_latency_ms:.2f}ms"
    )
    print(f"Reports written to {args.output}")


if __name__ == "__main__":
    main()
