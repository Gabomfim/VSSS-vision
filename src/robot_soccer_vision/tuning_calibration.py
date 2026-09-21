"""Interactive field-quad and ball bootstrap calibration for offline tuning."""

import cv2

from .app import _draw_label
from .field_rectifier import FieldRectifier

WINDOW = "Robot Soccer - Tuning Setup"


def calibrate_tuning_frame(
    frame, initial_radius: int = 17
) -> tuple[FieldRectifier, tuple[int, int], int]:
    rectifier = FieldRectifier()
    state = {"center": None, "error": ""}
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Ball radius", WINDOW, initial_radius, 100, lambda _value: None)

    def mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        if not rectifier.ready:
            if event == cv2.EVENT_LBUTTONDOWN:
                try:
                    rectifier.add_point((x, y))
                    state["error"] = ""
                except ValueError as error:
                    state["error"] = str(error)
            return
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            state["center"] = (x, y)

    cv2.setMouseCallback(WINDOW, mouse)
    try:
        while True:
            radius = max(2, cv2.getTrackbarPos("Ball radius", WINDOW))
            if not rectifier.ready:
                display = rectifier.draw_setup(frame)
                _draw_label(
                    display,
                    f"STEP 1/2 - FIELD QUAD: click {rectifier.next_corner_name}",
                    0,
                    (0, 255, 255),
                )
                _draw_label(
                    display,
                    "Order: top-left, top-right, bottom-right, bottom-left",
                    1,
                )
                _draw_label(display, "U: undo   R: restart   Q: cancel", 2)
                if state["error"]:
                    _draw_label(display, state["error"], 3, (0, 0, 255))
            else:
                display = rectifier.warp(frame)
                if state["center"] is None:
                    state["center"] = (display.shape[1] // 2, display.shape[0] // 2)
                cv2.circle(display, state["center"], radius, (255, 255, 255), 2)
                cv2.drawMarker(
                    display, state["center"], (255, 255, 255), cv2.MARKER_CROSS, 14, 1
                )
                _draw_label(
                    display,
                    "STEP 2/2 - BALL: place the circle over the orange ball",
                    0,
                    (0, 255, 255),
                )
                _draw_label(display, "Adjust Ball radius, then press SPACE", 1)
                _draw_label(display, "F: redo field quad   Q: cancel", 2)
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                raise RuntimeError("Tuning calibration was cancelled")
            if not rectifier.ready:
                if key == ord("u"):
                    rectifier.undo()
                    state["error"] = ""
                if key == ord("r"):
                    rectifier.reset()
                    state["error"] = ""
                continue
            if key == ord("f"):
                rectifier.reset()
                state["center"] = None
                continue
            if key == ord(" ") and state["center"] is not None:
                return rectifier, state["center"], radius
    finally:
        cv2.destroyWindow(WINDOW)


def first_video_frame(video_path: str):
    capture = cv2.VideoCapture(video_path)
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Could not read the first frame from: {video_path}")
    return frame

