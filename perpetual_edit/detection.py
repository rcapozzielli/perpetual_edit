"""Face Landmarker + Gesture Recognizer (MediaPipe Tasks) -> Features suavizadas."""
import math
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .models import ensure_model


@dataclass
class Features:
    face_found: bool = False
    face_bbox: tuple | None = None          # (x0, y0, x1, y1) normalizado 0-1
    face_landmarks: np.ndarray | None = None  # (478, 3) normalizado
    blendshapes: dict = field(default_factory=dict)  # suavizados
    yaw_deg: float = 0.0                    # + = virou para a direita da imagem
    pitch_deg: float = 0.0
    gesture: str = "None"
    gesture_score: float = 0.0
    hands: list = field(default_factory=list)  # lista de arrays (21, 3) normalizados
    face_missing_s: float = 0.0             # há quanto tempo não há rosto
    face_present_s: float = 0.0             # há quanto tempo o rosto está presente
    face_back_after_s: float = 0.0          # rosto acabou de voltar após N s fora (0 se não)
    aspect: float = 16 / 9                  # largura/altura do frame (corrige distâncias)
    custom: dict = field(default_factory=dict)  # saída dos modelos personalizados (training.py)


class EMA:
    """Média móvel exponencial por chave."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.values = {}

    def update(self, key, x):
        prev = self.values.get(key)
        v = x if prev is None else self.alpha * x + (1 - self.alpha) * prev
        self.values[key] = v
        return v

    def reset(self):
        self.values.clear()


class Detector:
    def __init__(self, detect_width=640, smoothing=0.5):
        self.detect_width = detect_width
        self.ema = EMA(smoothing)
        base = mp_python.BaseOptions
        mode = vision.RunningMode.VIDEO

        self.face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base(model_asset_path=ensure_model("face_landmarker.task")),
                running_mode=mode,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )
        self.gesture = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=base(model_asset_path=ensure_model("gesture_recognizer.task")),
                running_mode=mode,
                num_hands=2,
            )
        )
        self._last_ts = 0
        self._face_change_t = time.perf_counter()
        self._had_face = False
        self._ever_face = False   # o primeiro aparecimento não conta como "voltou"
        self._last_absence = 0.0

    def set_smoothing(self, alpha):
        self.ema.alpha = alpha

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = self.detect_width / w
        small = cv2.resize(frame_bgr, (self.detect_width, int(h * scale))) if scale < 1 else frame_bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # O modo VIDEO exige timestamps estritamente crescentes.
        ts = max(int(time.perf_counter() * 1000), self._last_ts + 1)
        self._last_ts = ts

        f = Features()
        face_res = self.face.detect_for_video(image, ts)
        if face_res.face_landmarks:
            lms = np.array([(p.x, p.y, p.z) for p in face_res.face_landmarks[0]], dtype=np.float32)
            f.face_found = True
            f.face_landmarks = lms
            x0, y0 = lms[:, 0].min(), lms[:, 1].min()
            x1, y1 = lms[:, 0].max(), lms[:, 1].max()
            f.face_bbox = tuple(self.ema.update(f"bbox{i}", float(v)) for i, v in enumerate((x0, y0, x1, y1)))
            if face_res.face_blendshapes:
                for c in face_res.face_blendshapes[0]:
                    f.blendshapes[c.category_name] = self.ema.update(c.category_name, c.score)
            if face_res.facial_transformation_matrixes:
                r = np.array(face_res.facial_transformation_matrixes[0])[:3, :3]
                yaw = math.degrees(math.atan2(r[0, 2], r[2, 2]))
                pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r[1, 2]))))
                f.yaw_deg = self.ema.update("yaw", yaw)
                f.pitch_deg = self.ema.update("pitch", pitch)
        else:
            # Sem rosto: zera a suavização para não "lembrar" do rosto anterior.
            self.ema.reset()

        now = time.perf_counter()
        f.aspect = w / h
        if f.face_found != self._had_face:
            if f.face_found and self._ever_face:
                self._last_absence = now - self._face_change_t
            self._had_face = f.face_found
            self._face_change_t = now
        if f.face_found:
            self._ever_face = True
            f.face_present_s = now - self._face_change_t
            if f.face_present_s < 1.0:  # janela de 1 s para o gatilho "rosto voltou"
                f.face_back_after_s = self._last_absence
        else:
            f.face_missing_s = now - self._face_change_t

        g_res = self.gesture.recognize_for_video(image, ts)
        if g_res.hand_landmarks:
            f.hands = [np.array([(p.x, p.y, p.z) for p in hand], dtype=np.float32) for hand in g_res.hand_landmarks]
        best = ("None", 0.0)
        for cats in g_res.gestures or []:
            if cats and cats[0].category_name != "None" and cats[0].score > best[1]:
                best = (cats[0].category_name, cats[0].score)
        f.gesture, f.gesture_score = best
        return f

    def close(self):
        self.face.close()
        self.gesture.close()
