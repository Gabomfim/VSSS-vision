"""Four-corner perspective calibration for a rectangular soccer field."""

from dataclasses import dataclass, field

import cv2
import numpy as np

CORNER_NAMES = ("TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT")


@dataclass(slots=True)
class FieldRectifier:
    points: list[tuple[float, float]] = field(default_factory=list)
    minimum_dimension: int = 80
    _matrix: np.ndarray | None = field(default=None, init=False, repr=False)
    _output_size: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _map1: np.ndarray | None = field(default=None, init=False, repr=False)
    _map2: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def ready(self) -> bool:
        return self._matrix is not None and self._output_size is not None

    @property
    def next_corner_name(self) -> str | None:
        return CORNER_NAMES[len(self.points)] if len(self.points) < 4 else None

    @property
    def output_size(self) -> tuple[int, int] | None:
        return self._output_size

    def add_point(self, point: tuple[int, int]) -> None:
        if self.ready or len(self.points) >= 4:
            return
        self.points.append((float(point[0]), float(point[1])))
        if len(self.points) == 4:
            self._compute()

    def undo(self) -> None:
        if self.points:
            self.points.pop()
        self._matrix = None
        self._output_size = None
        self._map1 = None
        self._map2 = None

    def reset(self) -> None:
        self.points.clear()
        self._matrix = None
        self._output_size = None
        self._map1 = None
        self._map2 = None

    def _compute(self) -> None:
        source = np.asarray(self.points, dtype=np.float32)
        contour = source.reshape((-1, 1, 2))
        area = abs(cv2.contourArea(contour))
        if area < self.minimum_dimension * self.minimum_dimension:
            self.undo()
            raise ValueError("Selected field is too small or its corners are incorrectly ordered")
        if not cv2.isContourConvex(contour.astype(np.int32)):
            self.undo()
            raise ValueError("Field corners must form a convex quadrilateral in the requested order")

        top = np.linalg.norm(source[1] - source[0])
        bottom = np.linalg.norm(source[2] - source[3])
        right = np.linalg.norm(source[2] - source[1])
        left = np.linalg.norm(source[3] - source[0])
        width = max(self.minimum_dimension, round(max(top, bottom)))
        height = max(self.minimum_dimension, round(max(left, right)))
        destination = np.asarray(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        self._matrix = cv2.getPerspectiveTransform(source, destination)
        self._output_size = (width, height)
        inverse = np.linalg.inv(self._matrix)
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
        homogeneous = np.stack(
            (grid_x.ravel(), grid_y.ravel(), np.ones(width * height, dtype=np.float32))
        )
        source_coordinates = inverse @ homogeneous
        source_coordinates /= source_coordinates[2:3]
        map_x = source_coordinates[0].reshape((height, width)).astype(np.float32)
        map_y = source_coordinates[1].reshape((height, width)).astype(np.float32)
        self._map1, self._map2 = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)

    def warp(self, frame: np.ndarray) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("Four field corners have not been selected")
        return cv2.remap(
            frame,
            self._map1,
            self._map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def draw_setup(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        integer_points = [(round(x), round(y)) for x, y in self.points]
        for index, point in enumerate(integer_points):
            cv2.circle(display, point, 6, (0, 255, 255), -1)
            cv2.putText(
                display,
                str(index + 1),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        for first, second in zip(integer_points, integer_points[1:]):
            cv2.line(display, first, second, (0, 255, 255), 2)
        if len(integer_points) == 4:
            cv2.line(display, integer_points[-1], integer_points[0], (0, 255, 255), 2)
        return display
