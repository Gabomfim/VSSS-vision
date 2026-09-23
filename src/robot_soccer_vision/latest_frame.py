"""Single-slot camera capture: old frames are discarded instead of queued."""

from threading import Condition, Thread
from time import monotonic

import cv2


class LatestFrameCapture:
    def __init__(
        self,
        source: int | str,
        start_paused_after_first: bool = False,
        manual_camera: bool = True,
        exposure: float | None = None,
    ) -> None:
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera or video source: {source}")
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if isinstance(source, int) and manual_camera:
            # OpenCV backends use different numeric conventions; these values request
            # manual exposure/white balance where the driver supports them.
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            if hasattr(cv2, "CAP_PROP_AUTO_WB"):
                self.capture.set(cv2.CAP_PROP_AUTO_WB, 0)
            if exposure is not None:
                self.capture.set(cv2.CAP_PROP_EXPOSURE, exposure)
        self._condition = Condition()
        self._frame = None
        self._sequence = 0
        self._captured_at = 0.0
        self._stopped = False
        self._paused = False
        self._pause_after_frame = start_paused_after_first
        source_fps = self.capture.get(cv2.CAP_PROP_FPS)
        self._playback_interval = (
            1.0 / source_fps
            if isinstance(source, str) and 1.0 <= source_fps <= 1000.0
            else 0.0
        )
        self._thread = Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while not self._stopped:
            with self._condition:
                self._condition.wait_for(lambda: not self._paused or self._stopped)
                if self._stopped:
                    break
            frame_started = monotonic()
            ok, frame = self.capture.read()
            with self._condition:
                if not ok:
                    self._stopped = True
                    self._condition.notify_all()
                    break
                self._frame = frame
                self._captured_at = monotonic()
                self._sequence += 1
                if self._pause_after_frame:
                    self._paused = True
                    self._pause_after_frame = False
                self._condition.notify_all()
                remaining = self._playback_interval - (monotonic() - frame_started)
                if remaining > 0 and not self._paused:
                    self._condition.wait(timeout=remaining)

    def pause(self) -> None:
        with self._condition:
            self._paused = True

    def resume(self) -> None:
        with self._condition:
            self._paused = False
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
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self.capture.release()
        self._thread.join(timeout=1.0)
