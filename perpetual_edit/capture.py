"""Captura da webcam numa thread, sempre expondo o frame mais recente.

Assim o loop principal nunca fica esperando o driver da câmera e
frames antigos são descartados em vez de acumular atraso.
"""
import os
import threading
import time

# Sem isso o MSMF pode levar vários segundos para abrir a câmera no Windows.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
import cv2  # noqa: E402

# No Windows, várias webcams integradas só entregam 720p a 10 fps via DirectShow,
# mas 30 fps via Media Foundation (msmf). Por isso msmf é o padrão.
BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}


class Camera:
    def __init__(self, index=0, width=1280, height=720, mirror=True, backend="msmf"):
        self.cap = cv2.VideoCapture(index, BACKENDS.get(backend, cv2.CAP_ANY))
        if not self.cap.isOpened():
            raise RuntimeError(f"Não consegui abrir a câmera {index} (backend {backend})")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.mirror = mirror
        self._frame = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._cond:
                self._frame = frame
                self._frame_id += 1
                self._cond.notify_all()

    def read(self, last_id=0, timeout=1.0):
        """Espera um frame mais novo que last_id. Devolve (id, frame) ou (last_id, None)."""
        with self._cond:
            self._cond.wait_for(lambda: self._frame_id > last_id or not self._running, timeout)
            if self._frame_id > last_id:
                return self._frame_id, self._frame
        return last_id, None

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()
