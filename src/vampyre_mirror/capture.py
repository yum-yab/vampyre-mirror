import shutil
import subprocess
import threading

import cv2
import numpy as np


class Camera:
    """Grabs webcam frames on a background thread and always serves the newest one.

    Stale frames are dropped instead of queued, which keeps the mirror latency low.
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        _disable_dynamic_framerate(index)

        self._frame: np.ndarray | None = None
        self._seq = 0
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def size(self) -> tuple[int, int]:
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self, last_seq: int = 0, timeout: float = 1.0) -> tuple[int, np.ndarray]:
        """Block until a frame newer than `last_seq` is available; return (seq, BGR frame)."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > last_seq, timeout):
                raise RuntimeError("Camera stopped delivering frames")
            return self._seq, self._frame

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


def _disable_dynamic_framerate(index: int) -> None:
    """Many UVC webcams drop to ~7 fps in dim light to expose longer; a mirror needs steady 30 fps."""
    if shutil.which("v4l2-ctl"):
        subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{index}", "-c", "exposure_dynamic_framerate=0"],
            capture_output=True,
        )
