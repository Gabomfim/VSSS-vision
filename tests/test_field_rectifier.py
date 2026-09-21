import cv2
import numpy as np
import pytest

from robot_soccer_vision.field_rectifier import FieldRectifier


def test_four_corners_create_rectified_field() -> None:
    rectifier = FieldRectifier(minimum_dimension=20)
    for point in ((40, 20), (180, 35), (200, 150), (20, 160)):
        rectifier.add_point(point)

    frame = np.zeros((190, 230, 3), dtype=np.uint8)
    cv2.circle(frame, (40, 20), 5, (0, 0, 255), -1)
    cv2.circle(frame, (180, 35), 5, (0, 255, 0), -1)
    warped = rectifier.warp(frame)

    assert rectifier.ready
    assert warped.shape[1] == rectifier.output_size[0]
    assert warped.shape[0] == rectifier.output_size[1]
    assert warped[0, 0, 2] > 100
    assert warped[0, -1, 1] > 100


def test_undo_and_reset_clear_calibration() -> None:
    rectifier = FieldRectifier(minimum_dimension=20)
    rectifier.add_point((10, 10))
    rectifier.add_point((100, 10))
    rectifier.undo()
    assert rectifier.points == [(10.0, 10.0)]

    rectifier.reset()
    assert rectifier.points == []
    assert not rectifier.ready


def test_crossed_corner_order_is_rejected() -> None:
    rectifier = FieldRectifier(minimum_dimension=20)
    rectifier.add_point((10, 10))
    rectifier.add_point((100, 100))
    rectifier.add_point((100, 10))

    with pytest.raises(ValueError, match="incorrectly ordered|convex"):
        rectifier.add_point((10, 100))

    assert not rectifier.ready
    assert len(rectifier.points) == 3
