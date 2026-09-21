"""Command-line entry point for self-supervised tuning and ablations."""

import argparse

from .tuning import tune_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune the expensive teacher, generate pseudo-labels, and benchmark fast trackers"
    )
    parser.add_argument("--source", required=True, help="Recorded calibration/match video")
    parser.add_argument("--initial-x", type=int, required=True, help="Ball center x in the first frame")
    parser.add_argument("--initial-y", type=int, required=True, help="Ball center y in the first frame")
    parser.add_argument("--radius", type=int, required=True, help="Approximate ball radius in pixels")
    parser.add_argument("--output", default="tuning-results", help="Report output directory")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--teacher-trials", type=int, default=18)
    args = parser.parse_args()

    teacher, ablations = tune_pipeline(
        args.source,
        (args.initial_x, args.initial_y),
        args.radius,
        args.output,
        args.max_frames,
        args.teacher_trials,
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

