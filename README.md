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

The fast version keeps only the newest camera frame, predicts a small search region with a constant-velocity Kalman filter, uses an 8-bit binary color mask and a precomputed fixed-radius circular kernel, and falls back to a half-resolution global search only after losing the ball. A recovery candidate is refined in a native-resolution local crop. Normal tracking does not use Hough transforms or multi-scale kernel banks. Its overlay reports processing time, capture-to-result latency, and whether ROI search was used.

During fast-tracker setup, the first frame is frozen while the field quad and ball parameters are selected. Acquisition begins only after **Space** confirms ball calibration. Pressing **C** or **F** pauses on the latest frame again. Recorded videos are paced using their encoded frame rate; live cameras continue to discard stale buffered frames.

For the lowest production-loop overhead, use a calibration report and disable rendering:

```bash
robot-soccer-track-fast --report tuning-results/report.json --no-display
```

Alternatively, retain a diagnostic preview but draw only one result out of every ten:

```bash
robot-soccer-track-fast --report tuning-results/report.json --display-every 10
```

Record an annotated tracking video with `--video-output`. Recording starts only after field
and ball calibration have finished, so setup frames are not included:

```bash
robot-soccer-track-fast \
  --source path/to/match.mp4 \
  --report tuning-results/report.json \
  --video-output output/tracking.mp4
```

The same option works without `--report`; in that case, complete the interactive field and
ball calibration first, and only the subsequent tracking frames are written. It can also be
combined with `--evaluate` and `--no-display` to generate an experiment video without the
preview-window rendering cost.

For live cameras, the fast tracker requests a one-frame backend buffer and manual exposure and white balance. `--exposure VALUE` passes a backend-specific exposure value to OpenCV. Use `--automatic-camera` only when automatic camera controls are intentionally required. On exit, the application prints mean, p95, and maximum capture-to-result latency; evaluation reports also store this distribution and the value for every frame.

Use another camera or a recorded match:

```bash
robot-soccer-track --source 1
robot-soccer-track --source path/to/match.mp4
```

## Calibrate and track

1. Click the four field corners in this order: **top-left**, **top-right**, **bottom-right**, **bottom-left**. Use **U** to undo a corner or **R** to restart the selection.
2. The camera image is rectified into a rectangular top-down field.
3. Move the pointer so the white overlay is centered on the orange ball.
4. Use **Ball radius** until the overlay follows the edge of the ball.
5. Press **Space** to sample its color and begin tracking.
6. Tune the color-tolerance sliders if lighting changes create misses or false matches.
7. Press **C**, or click in the video, to calibrate the ball again. Press **F** to redo the field corners. Press **Q** or **Esc** to quit.

Corner selection rejects crossed, non-convex, or implausibly small fields. The quad is used only to remove perspective skew; it does not infer physical dimensions, add margins, or convert coordinates to millimetres. The perspective mapping is converted into fixed remap tables once, so live frames do not recompute the homography. The fast tracker includes rectification time in its on-screen total latency.

In the adaptive version, the displayed HSV model updates only after a strong circle-and-size match. Press **A** to freeze or resume learning. A learning rate around 10–15% follows gradual daylight or exposure changes without reacting too strongly to one noisy frame.

## Self-supervised tuning and ablations

Record a representative video containing lighting changes, then run:

```bash
robot-soccer-tune \
  --source path/to/calibration.mp4 \
  --output tuning-results
```

The tuner opens an interactive setup on the first video frame:

1. Define the field as a four-point quad: top-left, top-right, bottom-right, bottom-left.
2. Review the rectified top-down view, place the circle overlay over the ball, and adjust the **Ball radius** slider.
3. Press **Space** to start tuning.

No initial position or radius command-line parameters are required. During tuning, the terminal displays a percentage bar and the active stage: video preparation, teacher search, pseudo-label generation, student ablations, and report generation.

The expensive teacher searches the initial HSV sample, HSV tolerances, radius tolerance, radius, and adaptation rate. It selects them using agreement between:

- forward and backward processing of the video;
- the original frames and synthetic brightness/saturation changes;
- multiple radii;
- component shape, disk-minus-ring convolution, and Hough-circle evidence.

For the final offline pseudo-labels, the teacher processes the complete recording in both directions. Forward and backward detections are associated only when their centers agree, fused using confidence weights, and passed through a constant-velocity Rauch-Tung-Striebel smoother. Short gaps are interpolated only when bounded by compatible detections; strong directional disagreements are rejected. The report records counts for fusion, rejection, unilateral evidence, and interpolation under `temporal_labeling`.

The student never receives future frames. Its velocity estimate and ROI use only detections already produced, so the deployed tracker remains strictly causal. The student ablation compares four preprocessing choices (`none`, circular opening, circular opening/closing, and Gaussian), two detectors (connected components and matched filtering), and ROI search on/off. Each row reports center error against the offline teacher, recall, mean latency, p95 latency, and a combined objective.

The output directory contains:

- `report.json`: selected teacher parameters and recommended fast pipeline;
- `ablations.csv`: all accuracy/latency comparisons;
- `pseudo_labels.csv`: bidirectionally fused and smoothed teacher centers and confidence values;
- `manifest.json`: input provenance, runtime information, Git revision, and SHA-256 hashes of every tuning artifact.

Use the selected pipeline directly:

```bash
robot-soccer-track-fast --report tuning-results/report.json
```

### Evaluate a calibrated fast tracker

To compare the report-backed fast tracker with the expensive adaptive teacher on the same incoming frames:

```bash
robot-soccer-track-fast \
  --source path/to/validation.mp4 \
  --report tuning-results/report.json \
  --evaluate \
  --evaluation-output tuning-results/fast-evaluation.json
```

The live overlay shows the current center difference and cumulative matched detections. The JSON summary is saved when the video ends or the user quits. It contains:

- fast and teacher detection rates;
- fast recall and miss rate relative to teacher detections;
- fast-only detection rate;
- mean, median, p95, and maximum center difference in pixels;
- agreement within one-quarter and one-half of the calibrated ball radius;
- fast and teacher latency distributions;
- teacher confidence distribution;
- source frames skipped by newest-frame acquisition.

Evaluation also creates `<name>.frames.csv`, containing the fast and teacher position, radius, confidence, score, latency, ROI usage, source sequence, skipped-frame count, and center error for every processed frame. `<name>.manifest.json` links the evaluation video or camera, calibration report, original calibration-video provenance, runtime, Git revision, and hashes of both evaluation artifacts.

Evaluation does not replace the fast tracker output with teacher output and does not adapt the fast tracker from teacher positions. Teacher agreement is a pseudo-label metric rather than independent ground truth, so the difficult-frame manual labels remain the appropriate check for shared systematic errors. `--evaluate` requires `--report`.

Pseudo-labels are not independent truth: teacher and student can share the same systematic error. For final validation, manually label a small, diverse holdout set (for example, 50–100 frames chosen across lighting conditions and field locations). This is much cheaper than adding position sensors and detects failure modes that self-consistency alone cannot expose.

## Label a random sample of difficult frames

After producing a tuning report, launch the active-learning labeler:

```bash
robot-soccer-label \
  --source path/to/calibration.mp4 \
  --report tuning-results/report.json \
  --count 100 \
  --pool-fraction 0.25 \
  --seed 7 \
  --output manual-labels.csv
```

The tool scores every candidate frame using:

- teacher ensemble confidence and vote count;
- teacher/student center disagreement;
- disagreement with the temporal motion prediction;
- Laplacian sharpness as a motion/defocus-blur signal;
- median brightness and shadow coverage.

It forms a pool from the most difficult 25% of frames by default, then samples the requested number uniformly and reproducibly from that pool. Increase `--pool-fraction` for more variety, or use `--stride 2` to score every second frame when scanning a long video.

In the labeling window:

- click the ball center;
- use `+` or `-` to adjust the circle radius;
- press **Enter** to save the label;
- press **N** when the ball is genuinely absent (a negative example);
- press **M** to skip motion blur;
- press **D** to skip darkness;
- press **O** to skip occlusion;
- press **S** for another impossible case;
- press **Q** to save progress and quit.

Every decision is written immediately to `manual-labels.csv`. Ball absence is stored as
`status=absent`, separately from impossible frames, and contributes a false-positive penalty
when `robot-soccer-refine` uses the active-learning labels. Running the same command again
safely resumes and excludes frames that were already reviewed. Skip reasons are preserved,
which makes it possible to analyze whether failures predominantly come from exposure, blur,
or occlusion.

The labeler additionally writes `manual-labels.selection.json` and `manual-labels.manifest.json`. These preserve every random sample, difficulty signal, seed, pool fraction, source-video hash, calibration-report hash, runtime, Git revision, reviewed counts, and artifact hashes.

## Reproducible report provenance

Tuning, evaluation, and manual validation outputs are designed as machine-readable inputs for a later LaTeX/PDF report. Provenance records include:

- absolute path, filename, byte size, modification time, and SHA-256 for every file input;
- the full command and working directory;
- UTC timestamp, host, operating system, Python, OpenCV, and NumPy versions;
- Git repository root, commit, dirty state, and tracked-diff hash;
- calibration parameters, selected quad, random seeds, sampling settings, and selected pipeline;
- output artifact paths and SHA-256 hashes.

The manifests intentionally do not hash themselves, avoiding recursive hashes. A live camera is recorded by its source identifier because it has no immutable file to hash; for a fully reproducible experiment, record the camera stream and evaluate that video file.

For the most consistent result, lock the camera exposure and white balance after the field lighting is set.

## Tests

The tests generate camera-like synthetic frames and check color, circle shape, and radius rejection:

```bash
pytest
```

## Coordinate convention

The origin is the top-left of the camera image: `x` grows to the right and `y` grows downward. A later field-calibration stage can transform these pixels into field coordinates.
