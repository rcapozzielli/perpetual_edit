"""Gatilhos personalizados: criação de modelos, coleta de amostras, treino e inferência.

Cada amostra é o rosto alinhado pelos olhos (128x128) + os blendshapes + a
posição das mãos naquele frame. As features (escolhidas por modelo) são:
  - "appearance": HOG do rosto alinhado em cinza (bordas: óculos, fone, garrafa, mão...)
  - "expression": os 52 blendshapes do MediaPipe
  - "hands":      onde estão as mãos em relação ao rosto (perto da boca? da orelha?)
"crop_scale" abre o recorte (1.0 = rosto justo; 1.5 = pega orelhas e mão chegando na boca).
O classificador é StandardScaler + LogisticRegression (scikit-learn), leve o
bastante para rodar em todo frame na CPU.

Arquivos: custom/<modelo>/<classe>/<sessao>_<n>.png (+ .npy blendshapes, _h.npy mãos)
          custom/<modelo>/model.joblib
          custom/models.json   modelos criados pelo app (tecla N no modo treino)
"""
import glob
import json
import math
import os
import re
import shutil
import time
import unicodedata

import cv2
import joblib
import numpy as np

from .config import ROOT

CUSTOM_DIR = os.path.join(ROOT, "custom")
USER_MODELS = os.path.join(CUSTOM_DIR, "models.json")
CROP = 128
_HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

# Cantos dos olhos no Face Landmarker (478 pontos)
_EYE_A = (33, 133)
_EYE_B = (362, 263)

# Padrões de um modelo criado pelo app: pensado para estados com mão/objeto
# perto do rosto (beber água, pôr fone, comer...).
NEW_MODEL_DEFAULTS = {
    "features": ["appearance", "expression", "hands"],
    "crop_scale": 1.5,
    "stable_prob": 0.75, "stable_frames": 8, "fresh_s": 2.0,
}


def align_face(frame, landmarks, size=CROP, scale=1.0):
    """Recorta o rosto com os olhos na horizontal, na mesma posição e escala.
    scale > 1 abre o recorte em volta do rosto (orelhas, mãos perto da boca)."""
    H, W = frame.shape[:2]
    pts = landmarks[:, :2] * (W, H)
    a, b = pts[list(_EYE_A)].mean(0), pts[list(_EYE_B)].mean(0)
    left, right = (a, b) if a[0] < b[0] else (b, a)
    center = (left + right) / 2
    dist = max(1.0, float(np.linalg.norm(right - left)))
    angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    m = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 0.4 * size / dist / scale)
    m[0, 2] += size / 2 - center[0]
    m[1, 2] += size * (0.5 - 0.1 / scale) - center[1]  # olhos a 40% da altura no recorte justo
    return cv2.warpAffine(frame, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- mãos
_HAND_PTS = (0, 4, 8, 12, 20)  # punho e pontas do polegar, indicador, médio, mínimo
_FACE_REFS = (13, 234, 454)    # boca, lateral esquerda e direita do rosto (orelhas)
_PER_HAND = 1 + 2 * len(_HAND_PTS) + len(_FACE_REFS)
HAND_DIM = 2 * _PER_HAND
_FAR = 3.0  # "distância" usada quando a mão não aparece (3 larguras de rosto)


def empty_hand_vector():
    v = np.zeros(HAND_DIM, np.float32)
    for k in range(2):
        v[k * _PER_HAND + 1 + 2 * len(_HAND_PTS):(k + 1) * _PER_HAND] = _FAR
    return v


def hand_vector(features):
    """Posição das mãos em relação ao rosto, em larguras de rosto (independe da distância
    da câmera): presença, punho/pontas dos dedos em relação ao nariz e a menor distância
    da mão até a boca e até cada lateral do rosto."""
    v = empty_hand_vector()
    lms = features.face_landmarks
    if lms is None or not features.hands:
        return v
    a = features.aspect  # x e y na mesma escala
    pts = lms[:, :2] * (a, 1.0)
    center = pts[4]
    scale = max(1e-3, float(pts[:, 0].max() - pts[:, 0].min()))
    refs = pts[list(_FACE_REFS)]
    for k, hand in enumerate(sorted(features.hands, key=lambda h: float(h[0, 0]))[:2]):
        hp = hand[:, :2] * (a, 1.0)
        off = k * _PER_HAND
        v[off] = 1.0
        v[off + 1:off + 1 + 2 * len(_HAND_PTS)] = ((hp[list(_HAND_PTS)] - center) / scale).ravel()
        d = np.linalg.norm(hp[None, :, :] - refs[:, None, :], axis=2).min(axis=1) / scale
        v[off + 1 + 2 * len(_HAND_PTS):off + _PER_HAND] = np.minimum(d, _FAR)
    return v


_GROUP_DIMS = {"appearance": 1764, "expression": 52, "hands": HAND_DIM}  # HOG 64x64 = 1764


def group_weights(feature_sets):
    """Peso por coluna que equilibra os grupos de features: sem isso os 1764 números
    do HOG "afogam" os 28 das mãos na regularização da regressão logística."""
    dims = [_GROUP_DIMS[g] for g in ("appearance", "expression", "hands") if g in feature_sets]
    biggest = max(dims)
    return np.concatenate([np.full(d, math.sqrt(biggest / d), np.float32) for d in dims])


def _weigh(X, w):  # no nível do módulo para o joblib conseguir salvar o pipeline
    return X * w


def extract(crop, blendshapes, feature_sets, bs_names, hands=None):
    parts = []
    if "appearance" in feature_sets:
        gray = _CLAHE.apply(cv2.cvtColor(cv2.resize(crop, (64, 64)), cv2.COLOR_BGR2GRAY))
        parts.append(_HOG.compute(gray).ravel())
    if "expression" in feature_sets:
        parts.append(np.array([blendshapes.get(n, 0.0) for n in bs_names], np.float32))
    if "hands" in feature_sets:
        parts.append(hands if hands is not None else empty_hand_vector())
    return np.concatenate(parts).astype(np.float32)


def model_dir(name):
    return os.path.join(CUSTOM_DIR, name)


def count_samples(name, classes):
    return {c: len(glob.glob(os.path.join(model_dir(name), c, "*.png"))) for c in classes}


def list_sessions(name, classes):
    """{classe: {ids das sessões de gravação}} — o id é o prefixo do nome do arquivo."""
    return {c: {os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(model_dir(name), c, "*.png"))}
            for c in classes}


def latest_session(name, classes):
    """(classe, sessão) da gravação mais recente — o id da sessão é a data/hora."""
    found = [(sid, c) for c, sids in list_sessions(name, classes).items() for sid in sids]
    if not found:
        return None
    sid, cls = max(found)
    return cls, sid


def trash_samples(name, cls, session=None):
    """Move as amostras de uma classe (ou só de uma sessão) para custom/_lixeira/.
    Nada é apagado: para desfazer, é só mover os arquivos de volta. Devolve quantas amostras."""
    src = os.path.join(model_dir(name), cls)
    pattern = f"{session}_*" if session else "*"
    files = [f for f in glob.glob(os.path.join(src, pattern)) if os.path.isfile(f)]
    if not files:
        return 0
    dst = os.path.join(CUSTOM_DIR, "_lixeira", name, cls, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dst, exist_ok=True)
    for f in files:
        shutil.move(f, os.path.join(dst, os.path.basename(f)))
    return sum(f.endswith(".png") for f in files)


def data_advice(counts, sessions):
    """Avisos sobre os dados que costumam causar disparo falso."""
    tips = []
    few = [c for c, n in sessions.items() if n < 2]
    if few:
        tips.append(f"só 1 sessão de {', '.join(few)}: grave mais, cada vez de um jeito "
                    "(falando, virando a cabeça, mão no rosto, outra luz)")
    lo, hi = min(counts, key=counts.get), max(counts, key=counts.get)
    if counts[hi] > 2 * counts[lo]:
        tips.append(f"'{hi}' tem {counts[hi]} amostras e '{lo}' só {counts[lo]}: equilibre")
    return tips


# ---------------------------------------------------------------- criar modelos
def sanitize(text):
    """'Bebendo água!' -> 'bebendo_agua' (nome seguro para pasta e config)."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]


def create_model(name, classes, path=None):
    """Cria o modelo em custom/models.json, já com um gatilho de transição
    classes[0] -> classes[1] que toca os templates. O config.json não é tocado."""
    path = path or USER_MODELS
    data = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    if name in data:
        raise ValueError(f"já existe um modelo '{name}'")
    data[name] = {
        "classes": classes, **NEW_MODEL_DEFAULTS,
        "trigger": {
            "enabled": True, "from": classes[0], "to": classes[1],
            "threshold": 0.7, "hold_frames": 3, "cooldown_s": 15,
            "label": name.replace("_", " "), "edits": ["@templates"],
        },
    }
    _write_models(data, path)
    return data[name]


def _write_models(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:  # sem BOM
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # troca atômica: nunca fica um models.json pela metade


def _read_models(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def is_user_model(name, path=None):
    """True se o modelo foi criado pelo app (está em custom/models.json)."""
    try:
        return name in _read_models(path or USER_MODELS)
    except (OSError, json.JSONDecodeError):
        return False


def remove_model(name, path=None):
    """Tira o modelo (e o gatilho dele) de custom/models.json e move a pasta inteira
    (amostras + model.joblib) para custom/_lixeira/<modelo>/modelo_inteiro_<data>/.
    Só vale para modelos criados pelo app: os do config.json ficam com você."""
    path = path or USER_MODELS
    data = _read_models(path)
    if name not in data:
        raise ValueError(f"'{name}' está declarado no config.json; apague de lá")
    dst = os.path.join(CUSTOM_DIR, "_lixeira", name, time.strftime("modelo_inteiro_%Y%m%d_%H%M%S"))
    os.makedirs(dst, exist_ok=True)
    src = model_dir(name)
    if os.path.isdir(src):
        for item in os.listdir(src):
            shutil.move(os.path.join(src, item), os.path.join(dst, item))
        os.rmdir(src)
    # guarda a entrada do models.json junto, para dar para restaurar à mão
    with open(os.path.join(dst, "models_entry.json"), "w", encoding="utf-8") as f:
        json.dump({name: data[name]}, f, ensure_ascii=False, indent=2)
    del data[name]
    _write_models(data, path)
    return dst


# ---------------------------------------------------------------- coleta
class SampleCollector:
    """Estado do modo treino: qual modelo, qual classe está gravando, contagens."""

    def __init__(self, custom_models):
        self.set_models(custom_models)
        self.recording = None  # nome da classe sendo gravada
        self.session = None
        self.n_frame = 0
        self.last_crop = None
        self.status = ""

    def set_models(self, custom_models):
        self.models = custom_models
        self.names = list(custom_models)
        self.idx = min(getattr(self, "idx", 0), max(0, len(self.names) - 1))
        self.refresh()

    def select(self, name):
        if name in self.names:
            self.stop()
            self.idx = self.names.index(name)
            self.refresh()

    def refresh(self):
        self._counts = count_samples(self.model, self.classes) if self.model else {}
        self._sessions = list_sessions(self.model, self.classes) if self.model else {}

    @property
    def model(self):
        return self.names[self.idx] if self.names else None

    @property
    def classes(self):
        return self.models[self.model]["classes"] if self.model else []

    @property
    def crop_scale(self):
        return self.models[self.model].get("crop_scale", 1.0) if self.model else 1.0

    def counts(self):
        return self._counts

    def sessions(self):
        return {c: len(v) for c, v in self._sessions.items()}

    def session_size(self, cls, session):
        return len(glob.glob(os.path.join(model_dir(self.model), cls, f"{session}_*.png")))

    def next_model(self):
        self.stop()
        if self.names:
            self.idx = (self.idx + 1) % len(self.names)
        self.refresh()

    def toggle(self, class_idx):
        if class_idx >= len(self.classes):
            return
        cls = self.classes[class_idx]
        if self.recording == cls:
            self.stop()
            return
        self.recording = cls
        self.session = time.strftime("%Y%m%d%H%M%S")
        self.n_frame = 0

    def stop(self):
        self.recording = None

    def capture(self, frame, features, every=2):
        """Chamado a cada frame no modo treino. Salva 1 amostra a cada `every` frames."""
        if features.face_landmarks is None:
            self.last_crop = None
            return
        self.last_crop = align_face(frame, features.face_landmarks, scale=self.crop_scale)
        if not self.recording:
            return
        self.n_frame += 1
        if self.n_frame % every:
            return
        d = os.path.join(model_dir(self.model), self.recording)
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, f"{self.session}_{self.n_frame:05d}")
        cv2.imwrite(base + ".png", self.last_crop)
        np.save(base + ".npy", np.array([features.blendshapes.get(n, 0.0) for n in BS_NAMES], np.float32))
        np.save(base + "_h.npy", hand_vector(features))
        self._counts[self.recording] = self._counts.get(self.recording, 0) + 1
        self._sessions.setdefault(self.recording, set()).add(self.session)


# Ordem fixa dos 52 blendshapes do MediaPipe (o .npy de cada amostra segue esta ordem)
BS_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]


# ---------------------------------------------------------------- treino
def train_model(name, model_cfg, min_per_class=10):
    """Treina e salva custom/<name>/model.joblib. Devolve uma mensagem de resultado."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    classes = model_cfg["classes"]
    feats = model_cfg.get("features", ["appearance", "expression"])
    X, y, groups = [], [], []
    for ci, cls in enumerate(classes):
        for png in sorted(glob.glob(os.path.join(model_dir(name), cls, "*.png"))):
            crop = cv2.imread(png)
            npy, hnpy = png[:-4] + ".npy", png[:-4] + "_h.npy"
            bs = dict(zip(BS_NAMES, np.load(npy))) if os.path.exists(npy) else {}
            hands = np.load(hnpy) if os.path.exists(hnpy) else None  # amostras antigas: "sem mão"
            X.append(extract(crop, bs, feats, BS_NAMES, hands))
            y.append(ci)
            groups.append(f"{cls}/{os.path.basename(png).split('_')[0]}")  # sessão de gravação
    y = np.array(y)
    counts = {c: int((y == i).sum()) for i, c in enumerate(classes)}
    if len(classes) < 2 or min(counts.values()) < min_per_class:
        return f"poucas amostras {counts} (mín. {min_per_class} por classe)"
    X = np.stack(X)

    weights = group_weights(feats)

    def make():
        return make_pipeline(StandardScaler(), FunctionTransformer(_weigh, kw_args={"w": weights}),
                             LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"))

    # Validação separando por sessão: frames da mesma gravação nunca ficam
    # ao mesmo tempo no treino e no teste (senão a acurácia fica inflada).
    sessions_per_class = {c: len({g for g, yy in zip(groups, y) if yy == i}) for i, c in enumerate(classes)}
    if min(sessions_per_class.values()) >= 2:
        n_splits = min(5, min(sessions_per_class.values()))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
        scores = cross_val_score(make(), X, y, groups=groups, cv=cv)
        val = f"validação {scores.mean() * 100:.0f}% ({n_splits} folds por sessão)"
    else:
        val = "grave 2+ sessões por classe para ver a validação"

    clf = make().fit(X, y)
    joblib.dump({"classes": classes, "features": feats, "bs_names": BS_NAMES, "pipeline": clf,
                 "crop_scale": model_cfg.get("crop_scale", 1.0)},
                os.path.join(model_dir(name), "model.joblib"))
    tips = data_advice(counts, sessions_per_class)
    msg = f"'{name}' treinado com {sum(counts.values())} amostras; {val}"
    return msg + "".join(f". ATENÇÃO: {t}" for t in tips)


# ---------------------------------------------------------------- inferência
class CustomModels:
    """Roda os modelos treinados a cada frame e mantém um "estado estável" por modelo."""

    def __init__(self, custom_models, smoothing=0.5):
        self.models = {}
        self.load(custom_models, smoothing)

    def load(self, custom_models, smoothing=0.5):
        self.alpha = smoothing
        self.cfg = custom_models
        self.models, missing = {}, []
        for name in custom_models:
            path = os.path.join(model_dir(name), "model.joblib")
            if not os.path.exists(path):
                missing.append(name)
                continue
            m = joblib.load(path)
            m.setdefault("crop_scale", 1.0)  # modelos treinados antes desta opção
            m["state"] = {"probs": {c: 0.0 for c in m["classes"]}, "stable": None, "previous": None,
                          "changed_at": 0.0, "count": 0, "top": None}
            self.models[name] = m
        return missing

    def update(self, frame, features):
        """Preenche features.custom[modelo] = {probs, stable, previous, since, fresh_s}."""
        now = time.perf_counter()
        crops, hands = {}, None  # recortes por escala, calculados uma vez por frame
        for name, m in self.models.items():
            mcfg = self.cfg.get(name, {})
            st = m["state"]
            if features.face_landmarks is not None:
                scale = m["crop_scale"]
                if scale not in crops:
                    crops[scale] = align_face(frame, features.face_landmarks, scale=scale)
                if hands is None and "hands" in m["features"]:
                    hands = hand_vector(features)
                x = extract(crops[scale], features.blendshapes, m["features"], m["bs_names"], hands)
                p = m["pipeline"].predict_proba(x[None])[0]
                for c, pv in zip(m["classes"], p):
                    st["probs"][c] = self.alpha * float(pv) + (1 - self.alpha) * st["probs"][c]
                top = max(st["probs"], key=st["probs"].get)
                if st["probs"][top] >= mcfg.get("stable_prob", 0.75):
                    st["count"] = st["count"] + 1 if top == st["top"] else 1
                    st["top"] = top
                else:
                    st["count"], st["top"] = 0, None
                if st["top"] and st["top"] != st["stable"] and st["count"] >= mcfg.get("stable_frames", 8):
                    st["previous"], st["stable"], st["changed_at"] = st["stable"], st["top"], now
            features.custom[name] = {
                "probs": dict(st["probs"]), "stable": st["stable"], "previous": st["previous"],
                "since": now - st["changed_at"], "fresh_s": mcfg.get("fresh_s", 2.0),
            }
