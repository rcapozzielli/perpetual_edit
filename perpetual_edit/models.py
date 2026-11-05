"""Download dos modelos .task do MediaPipe na primeira execução."""
import os
import urllib.request

from .config import ROOT

MODELS_DIR = os.path.join(ROOT, "models")

MODEL_URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
}


def ensure_model(name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    print(f"[models] baixando {name} ...")
    tmp = path + ".part"
    urllib.request.urlretrieve(MODEL_URLS[name], tmp)
    os.replace(tmp, path)
    print(f"[models] ok ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path
