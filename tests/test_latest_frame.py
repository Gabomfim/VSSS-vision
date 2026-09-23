from time import sleep

import numpy as np

import robot_soccer_vision.latest_frame as latest_frame_module
from robot_soccer_vision.latest_frame import LatestFrameCapture


class FakeVideoCapture:
    def __init__(self, _source) -> None:
        self.index = 0
        self.released = False
        self.settings = []

    def isOpened(self) -> bool:
        return True

    def set(self, _property, _value) -> bool:
        self.settings.append((_property, _value))
        return True

    def get(self, _property) -> float:
        return 30.0

    def read(self):
        if self.released or self.index >= 10:
            return False, None
        frame = np.full((8, 8, 3), self.index, dtype=np.uint8)
        self.index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def test_capture_freezes_first_frame_until_resumed(monkeypatch) -> None:
    fake = FakeVideoCapture("unused")
    monkeypatch.setattr(latest_frame_module.cv2, "VideoCapture", lambda _source: fake)
    capture = LatestFrameCapture("recording.mp4", start_paused_after_first=True)
    try:
        ok, frame, sequence, _ = capture.read(timeout=0.2)
        assert ok and sequence == 1
        assert np.all(frame == 0)

        sleep(0.03)
        advanced, _, same_sequence, _ = capture.read(sequence, timeout=0.02)
        assert not advanced
        assert same_sequence == sequence
        assert fake.index == 1

        capture.resume()
        advanced, next_frame, next_sequence, _ = capture.read(sequence, timeout=0.2)
        assert advanced and next_sequence > sequence
        assert np.all(next_frame == 1)
    finally:
        capture.release()


def test_live_camera_requests_single_buffer_and_manual_controls(monkeypatch) -> None:
    fake = FakeVideoCapture(0)
    monkeypatch.setattr(latest_frame_module.cv2, "VideoCapture", lambda _source: fake)
    capture = LatestFrameCapture(0, manual_camera=True, exposure=-7.0)
    try:
        assert (latest_frame_module.cv2.CAP_PROP_BUFFERSIZE, 1) in fake.settings
        assert (latest_frame_module.cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) in fake.settings
        assert (latest_frame_module.cv2.CAP_PROP_EXPOSURE, -7.0) in fake.settings
    finally:
        capture.release()
