# Robot Soccer Vision

An overhead-camera ball tracker for robot soccer. The first implementation combines a picked HSV color with a fixed-size circle template, which is a good fit for a stable camera facing a flat field.

Three live versions are included: a fixed-color tracker, an adaptive tracker that learns gradual lighting changes, and a tuned low-latency tracker. An offline teacher/student pipeline can learn parameters and compare preprocessing/detection ablations from recorded video without external position sensors.

## What it does

- Shows the live camera feed.
- Lets you place a circle overlay over the ball and adjust its radius.
- Samples the ball color from inside that circle.
- Finds regions with the selected color, then scores them against the expected circle shape and size.
- Displays the ball center `(x, y)`, measured radius, and confidence score.

## Setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[dev]"
```

## Run

Use the default camera:

```bash
robot-soccer-track
```

Run the adaptive-lighting version:

```bash
robot-soccer-track-adaptive
```

Run the latency-optimized version:

```bash
robot-soccer-track-fast
```

The fast version keeps only the newest camera frame, predicts a small search region from ball velocity, uses a precomputed fixed-radius circular kernel, and falls back to a half-resolution global search after losing the ball. Its overlay reports processing time, camera-frame age, and whether ROI search was used.

Use another camera or a recorded match:

```bash
robot-soccer-track --source 1
robot-soccer-track --source path/to/match.mp4
```

## Calibrate and track

1. Move the pointer so the white overlay is centered on the orange ball.
2. Use **Ball radius** until the overlay follows the edge of the ball.
3. Press **Space** to sample its color and begin tracking.
4. Tune the color-tolerance sliders if lighting changes create misses or false matches.
5. Press **C**, or click in the video, to calibrate again. Press **Q** or **Esc** to quit.

In the adaptive version, the displayed HSV model updates only after a strong circle-and-size match. Press **A** to freeze or resume learning. A learning rate around 10–15% follows gradual daylight or exposure changes without reacting too strongly to one noisy frame.

## Self-supervised tuning and ablations

Record a representative video containing lighting changes and note the ball center and radius in its first frame. Then run:

```bash
robot-soccer-tune \
  --source path/to/calibration.mp4 \
  --initial-x 640 \
  --initial-y 360 \
  --radius 14 \
  --output tuning-results
```

The expensive teacher searches the initial HSV sample, HSV tolerances, radius tolerance, radius, and adaptation rate. It selects them using agreement between:

- forward and backward processing of the video;
- the original frames and synthetic brightness/saturation changes;
- multiple radii;
- component shape, disk-minus-ring convolution, and Hough-circle evidence.

Only consensus detections become pseudo-labels. The student ablation compares four preprocessing choices (`none`, circular opening, circular opening/closing, and Gaussian), two detectors (connected components and matched filtering), and ROI search on/off. Each row reports center error against the teacher, recall, mean latency, p95 latency, and a combined objective.

The output directory contains:

- `report.json`: selected teacher parameters and recommended fast pipeline;
- `ablations.csv`: all accuracy/latency comparisons;
- `pseudo_labels.csv`: teacher centers and confidence values.

Use the selected pipeline directly:

```bash
robot-soccer-track-fast --report tuning-results/report.json
```

Pseudo-labels are not independent truth: teacher and student can share the same systematic error. For final validation, manually label a small, diverse holdout set (for example, 50–100 frames chosen across lighting conditions and field locations). This is much cheaper than adding position sensors and detects failure modes that self-consistency alone cannot expose.

For the most consistent result, lock the camera exposure and white balance after the field lighting is set.

## Tests

The tests generate camera-like synthetic frames and check color, circle shape, and radius rejection:

```bash
pytest
```

## Coordinate convention

The origin is the top-left of the camera image: `x` grows to the right and `y` grows downward. A later field-calibration stage can transform these pixels into field coordinates.
