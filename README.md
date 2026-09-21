# Robot Soccer Vision

An overhead-camera ball tracker for robot soccer. The first implementation combines a picked HSV color with a fixed-size circle template, which is a good fit for a stable camera facing a flat field.

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

For the most consistent result, lock the camera exposure and white balance after the field lighting is set.

## Tests

The tests generate camera-like synthetic frames and check color, circle shape, and radius rejection:

```bash
pytest
```

## Coordinate convention

The origin is the top-left of the camera image: `x` grows to the right and `y` grows downward. A later field-calibration stage can transform these pixels into field coordinates.
