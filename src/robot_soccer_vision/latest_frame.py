"""Single-slot camera capture: old frames are discarded instead of queued."""

from threading import Condition, Thread
from time import monotonic

import cv2


class LatestFrameCapture:
    def __init__(self, source: int | str) -> None:
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera or video source: {source}")
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._condition = Condition()
        self._frame = None
        self._sequence = 0
        self._captured_at = 0.0
        self._stopped = False
        self._thread = Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while not self._stopped:
            ok, frame = self.capture.read()
            with self._condition:
                if not ok:
                    self._stopped = True
                    self._condition.notify_all()
                    break
                self._frame = frame
                self._captured_at = monotonic()
                self._sequence += 1
                self._condition.notify_all()

    def read(self, after_sequence: int = 0, timeout: float = 1.0):
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > after_sequence or self._stopped, timeout=timeout
            )
            if self._sequence <= after_sequence:
                return False, None, self._sequence, self._captured_at
            return True, self._frame.copy(), self._sequence, self._captured_at

    @property
    def finished(self) -> bool:
        return self._stopped

    def release(self) -> None:
        self._stopped = True
        self.capture.release()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=1.0)
