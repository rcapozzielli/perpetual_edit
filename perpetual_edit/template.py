"""Templates: recria um edit (ex.: template do CapCut) usando a SUA imagem.

Um template é um JSON com a música, a duração e uma lista de trechos. Cada
trecho tem uma origem de imagem e uma lista de efeitos:
  - "replay": a câmera gravada de `before_s` antes do gatilho até o edit começar
              (inclui o tempo de "loading"). Cada trecho escolhe o momento com
              "replay_at" (segundos relativos ao gatilho) e a velocidade com "speed".
  - "live":   a câmera ao vivo
  - "video":  frames de um vídeo (ex.: a prévia exportada do CapCut)
  - "keyed":  TELA VERDE. O vídeo do template (exportado do CapCut com os espaços
              preenchidos de verde) é a base; só os pixels verdes viram o seu replay.
              Textos, emojis e transições do template continuam originais.
Efeitos: fade_in, fade_out, vignette, zoom, whip, shake, face_sticker.

A imagem é sempre enquadrada no formato do template (ex.: 3:4) e centrada no
seu rosto. Os landmarks acompanham cada frame (inclusive no replay), então
adesivos colam no rosto mesmo quando o trecho é uma repetição.
"""
import collections
import json
import math
import os
import random
import threading
import time

import cv2
import numpy as np

from .audio import extract_audio
from .config import resolve

EMOJI_FONT = r"C:\Windows\Fonts\seguiemj.ttf"
LANDMARKS = {"nose": 4, "nose_bridge": 197, "between_eyes": 168, "chin": 152, "forehead": 10, "mouth": 13}


def load_template(path):
    with open(path, "r", encoding="utf-8-sig") as f:  # -sig: aceita arquivo salvo com BOM
        spec = json.load(f)
    base = os.path.dirname(path)
    for seg in spec["segments"]:
        if seg["source"] == "video":
            seg["file"] = resolve(seg.get("file", spec.get("video")))
    if spec.get("video"):
        spec["video"] = resolve(spec["video"])
    audio = spec.get("audio", "video")
    spec["audio_file"] = spec["video"] if audio == "video" else resolve(audio) or os.path.join(base, audio)
    if spec["audio_file"] and not os.path.exists(spec["audio_file"]):
        raise FileNotFoundError(f"áudio do template não encontrado: {spec['audio_file']}")
    spec["size"] = tuple(int(v) for v in spec.get("size", (540, 720)))
    spec["duration"] = float(spec.get("duration", max(s["end"] for s in spec["segments"])))
    for seg in spec["segments"]:
        if seg["source"] == "video" and not (seg["file"] and os.path.exists(seg["file"])):
            raise FileNotFoundError(f"vídeo do template não encontrado: {seg.get('file')}")
    if any(s["source"] == "keyed" for s in spec["segments"]):
        if not (spec.get("video") and os.path.exists(spec["video"])):
            raise FileNotFoundError(f"vídeo de tela verde não encontrado: {spec.get('video')}")
        spec["key"] = {**KEY_DEFAULT, **spec.get("key", {})}
    return spec


# ---------------------------------------------------------------- tela verde (chroma key)
# Faixa de verde em HSV do OpenCV (hue 0-180). Saturação mínima alta separa o verde de
# cinzas e pretos; brilho mínimo baixo para pegar o verde mesmo quando o template escurece.
KEY_DEFAULT = {"hue": [35, 88], "sat_min": 80, "val_min": 25, "ref_value": 230, "light": 0.6}
_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def key_mask(frame, key=None):
    """Máscara 0-1 (float32) dos pixels verdes, com a borda suavizada, + o HSV do quadro."""
    k = {**KEY_DEFAULT, **(key or {})}
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (k["hue"][0], k["sat_min"], k["val_min"]), (k["hue"][1], 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _K3)   # tira pontinhos soltos (ruído da compressão)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _K5)  # fecha buraquinhos dentro do verde
    return m, hsv


def light_map(hsv, hard, ref_value):
    """Quanto o template clareou/escureceu o verde em cada ponto (1 = igual ao slot original).
    Copia para a sua imagem fades, flashes, vinhetas e piscadas do template. É calculado
    em baixa resolução e "fechado" (máximo local) para ignorar o número escuro do slot."""
    h, w = hard.shape
    sw = max(16, w // 6)
    sh = max(16, round(sw * h / w))
    v = cv2.resize(np.where(hard > 0, hsv[:, :, 2], 0).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_AREA)
    m = cv2.resize(hard, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    v = v.astype(np.float32) / np.maximum(m, 0.05)  # média só dos pixels verdes
    v = cv2.morphologyEx(v, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    v = cv2.GaussianBlur(v, (0, 0), 1.0)
    return np.clip(cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR) / ref_value, 0.0, 1.3)


def compose_keyed(base, user_at, key=None):
    """Troca o verde de `base` pela sua imagem.
    user_at(size) devolve a sua imagem (BGR) no tamanho (w, h) da caixa que o verde ocupa;
    assim o zoom e o deslize que o template aplica no espaço movem você junto."""
    k = {**KEY_DEFAULT, **(key or {})}
    hard, hsv = key_mask(base, k)
    if cv2.countNonZero(hard) < 0.003 * hard.size:  # sem verde: quadro original do template
        return base
    x, y, bw, bh = cv2.boundingRect(hard)
    if bw < 8 or bh < 8:
        return base
    H, W = base.shape[:2]
    user = user_at((bw, bh))
    user = cv2.copyMakeBorder(user, y, H - y - bh, x, W - x - bw, cv2.BORDER_REPLICATE)
    # Luz: o filtro do template também puxa o verde para baixo por ser um ajuste de COR,
    # que escureceria demais a sua pele. "light" < 1 copia só parte do escurecimento.
    gain = light_map(hsv, hard, k["ref_value"]) ** k.get("light", 0.6)
    user = cv2.multiply(user, cv2.merge([gain] * 3), dtype=cv2.CV_8U)
    a = cv2.GaussianBlur(hard, (0, 0), 1.2).astype(np.float32) / 255
    # despill: perto do verde (borda de texto e emoji), tira o reflexo verde da compressão
    b, g, r = cv2.split(base)
    g = np.where(a > 0, cv2.min(g, cv2.max(b, r)), g)
    base = cv2.merge([b, g, r])
    return cv2.blendLinear(user, base, a, 1.0 - a)


# ---------------------------------------------------------------- enquadramento + replay
class Framer:
    """Recorta a câmera no formato do template, centrado no rosto (suavizado)."""

    def __init__(self, size=(540, 720), face_height=0.42, smoothing=0.15):
        self.size = size
        self.face_height = face_height  # fração da altura do recorte ocupada pelo rosto
        self.alpha = smoothing
        self.box = None  # (cx, cy, h) em pixels da câmera

    def crop(self, frame, features):
        H, W = frame.shape[:2]
        ow, oh = self.size
        aspect = ow / oh
        if features.face_found and features.face_bbox:
            x0, y0, x1, y1 = features.face_bbox
            target = ((x0 + x1) / 2 * W, (y0 + y1) / 2 * H, (y1 - y0) * H / self.face_height)
        elif self.box is None:
            target = (W / 2, H / 2, H)
        else:
            target = self.box
        self.box = target if self.box is None else tuple(
            self.alpha * t + (1 - self.alpha) * b for t, b in zip(target, self.box))
        cx, cy, h = self.box
        h = float(np.clip(h, H * 0.4, H))
        w = h * aspect
        if w > W:
            w, h = W, W / aspect
        x = float(np.clip(cx - w / 2, 0, W - w))
        y = float(np.clip(cy - h / 2, 0, H - h))
        crop = cv2.resize(frame[int(y):int(y + h), int(x):int(x + w)], (ow, oh), interpolation=cv2.INTER_AREA)
        lms = None
        if features.face_landmarks is not None:
            lms = features.face_landmarks[:, :2].copy()
            lms[:, 0] = (lms[:, 0] * W - x) / w
            lms[:, 1] = (lms[:, 1] * H - y) / h
        return crop, lms


def fit_aspect(img, lms, size):
    """Recorta o centro de `img` no formato de `size` (sem distorcer) e redimensiona.
    O replay é gravado quadrado; cada template tira dele o seu formato (3:4, 9:16, 1:1...)."""
    h, w = img.shape[:2]
    target = size[0] / size[1]
    if abs(w / h - target) > 1e-3:
        if w / h > target:  # sobra largura
            cw, ch = int(round(h * target)), h
        else:  # sobra altura
            cw, ch = w, int(round(w / target))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        img = img[y0:y0 + ch, x0:x0 + cw]
        if lms is not None:
            lms = lms.copy()
            lms[:, 0] = (lms[:, 0] * w - x0) / cw
            lms[:, 1] = (lms[:, 1] * h - y0) / ch
    if img.shape[1::-1] != tuple(size):
        img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_LINEAR)
    return img, lms


class ReplayBuffer:
    """Guarda os últimos `seconds` de câmera já enquadrados (+ landmarks).
    Quadrado (600x600) para servir templates de qualquer formato via fit_aspect."""

    def __init__(self, seconds=3.0, size=(600, 600)):
        self.seconds = seconds
        self.framer = Framer(size)
        self.frames = collections.deque()

    def push(self, frame, features, t=None):
        t = time.perf_counter() if t is None else t
        crop, lms = self.framer.crop(frame, features)
        self.frames.append((t, crop, lms))
        while self.frames and t - self.frames[0][0] > self.seconds:
            self.frames.popleft()

    def latest(self):
        return self.frames[-1] if self.frames else (0.0, np.zeros((*self.framer.size[::-1], 3), np.uint8), None)

    def window(self, t_start, t_end, t_ref):
        """Frames entre t_start e t_end, com o tempo relativo a t_ref (o instante do gatilho)."""
        return [(t - t_ref, c, l) for t, c, l in self.frames if t_start <= t <= t_end]


# ---------------------------------------------------------------- efeitos
_vignette_cache = {}
_emoji_cache = {}


def _ease_out(x):
    return 1 - (1 - x) ** 3


def _span(t, dur, p):
    """Progresso 0-1 do efeito dentro do trecho (respeita "at" e "duration")."""
    at = p.get("at", 0.0)
    d = p.get("duration", dur - at)
    return None if t < at or t > at + d else (t - at) / max(1e-6, d)


def _warp(img, lms, m, border=cv2.BORDER_REFLECT):
    h, w = img.shape[:2]
    img = cv2.warpAffine(img, m, (w, h), borderMode=border)  # BORDER_CONSTANT = preto
    if lms is not None:
        px = np.c_[lms * (w, h), np.ones(len(lms))] @ m.T
        lms = px / (w, h)
    return img, lms


def fx_fade_in(img, lms, t, dur, p):
    a = min(1.0, t / max(1e-6, p.get("duration", 0.5)))
    return (img if a >= 1 else cv2.convertScaleAbs(img, alpha=a)), lms


def fx_fade_out(img, lms, t, dur, p):
    a = min(1.0, (dur - t) / max(1e-6, p.get("duration", 0.5)))
    return (img if a >= 1 else cv2.convertScaleAbs(img, alpha=max(0.0, a))), lms


def fx_mirror(img, lms, t, dur, p):
    """Espelha a imagem na horizontal (os landmarks acompanham)."""
    if _span(t, dur, p) is None:
        return img, lms
    if lms is not None:
        lms = lms.copy()
        lms[:, 0] = 1.0 - lms[:, 0]
    return cv2.flip(img, 1), lms


def fx_flash(img, lms, t, dur, p):
    """Flash branco no início do trecho que se apaga (transição clássica de edit)."""
    k = _span(t, dur, p)
    if k is None:
        return img, lms
    a = p.get("strength", 0.85) * (1 - k) ** 2
    return cv2.addWeighted(img, 1 - a, np.full_like(img, 255), a, 0), lms


def fx_vignette(img, lms, t, dur, p):
    h, w = img.shape[:2]
    key = (w, h, p.get("strength", 0.6))
    if key not in _vignette_cache:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / math.sqrt(2)
        mask = 1 - key[2] * np.clip((r - 0.35) / 0.65, 0, 1) ** 1.5
        _vignette_cache[key] = cv2.merge([mask] * 3)
    return cv2.multiply(img, _vignette_cache[key], dtype=cv2.CV_8U), lms


def fx_zoom(img, lms, t, dur, p):
    k = _span(t, dur, p)
    if k is None:
        k = 1.0 if t > p.get("at", 0.0) else 0.0
    s = p.get("from", 1.0) + (p.get("to", 1.2) - p.get("from", 1.0)) * (_ease_out(k) if p.get("ease") == "out" else k)
    h, w = img.shape[:2]
    c = (w / 2, h / 2)
    if p.get("anchor") == "face" and lms is not None:
        c = tuple(lms[LANDMARKS["nose"]] * (w, h))
    return _warp(img, lms, cv2.getRotationMatrix2D(c, 0, s))


_DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def fx_whip(img, lms, t, dur, p):
    """Chicote: deslocamento rápido com borrão de movimento.
    mode "in" (padrão): entra desacelerando; "out": sai acelerando (para o fim do trecho)."""
    k = _span(t, dur, p)
    if k is None:
        return img, lms
    h, w = img.shape[:2]
    amount = 1 - _ease_out(k) if p.get("mode", "in") == "in" else k ** 2
    dx, dy = _DIRS[p.get("direction", "up")]
    size = w if dx else h
    shift = p.get("distance", 0.6) * size * amount
    m = np.float32([[1, 0, dx * shift], [0, 1, dy * shift]])
    # saindo, o espaço que sobra fica preto; entrando, reflete a imagem (esconde a borda)
    img, lms = _warp(img, lms, m, cv2.BORDER_CONSTANT if p.get("mode") == "out" else cv2.BORDER_REFLECT)
    blur = int(p.get("blur", 0.25) * size * amount) | 1
    if blur > 1:
        kernel = np.full((1, blur) if dx else (blur, 1), 1.0 / blur, np.float32)
        img = cv2.filter2D(img, -1, kernel)
    return img, lms


def fx_shake(img, lms, t, dur, p):
    """Balanço de câmera na mão: translação + rotação suaves que diminuem."""
    k = _span(t, dur, p)
    if k is None:
        return img, lms
    h, w = img.shape[:2]
    amp = 1 - k
    f = p.get("freq", 2.5)
    ang = p.get("rotate", 4.0) * amp * math.sin(2 * math.pi * f * t)
    dx = p.get("amount", 0.03) * w * amp * math.sin(2 * math.pi * f * 1.3 * t + 1.0)
    dy = p.get("amount", 0.03) * h * amp * math.cos(2 * math.pi * f * 0.9 * t)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0 + 0.06 * amp)  # zoom leve esconde as bordas
    m[:, 2] += (dx, dy)
    return _warp(img, lms, m)


def _emoji(char, px):
    key = (char, px)
    if key not in _emoji_cache:
        from PIL import Image, ImageDraw, ImageFont
        size = 109  # a fonte de emoji do Windows renderiza bem nesse tamanho
        im = Image.new("RGBA", (size * 2, size * 2))
        ImageDraw.Draw(im).text((size // 4, size // 4), char, font=ImageFont.truetype(EMOJI_FONT, size),
                                embedded_color=True)
        im = im.crop(im.getbbox())
        im = im.resize((px, max(1, round(px * im.height / im.width))), Image.LANCZOS)
        rgba = np.array(im)
        _emoji_cache[key] = (cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR), rgba[:, :, 3:] / 255.0)
    return _emoji_cache[key]


def fx_face_sticker(img, lms, t, dur, p):
    """Cola um emoji num ponto do rosto, com tamanho proporcional à largura do rosto."""
    if lms is None or _span(t, dur, p) is None:
        return img, lms
    h, w = img.shape[:2]
    face_w = (lms[:, 0].max() - lms[:, 0].min()) * w
    px = max(8, int(face_w * p.get("scale", 0.15)))
    bgr, alpha = _emoji(p.get("emoji", "💔"), px)
    idx = LANDMARKS.get(p.get("landmark", "nose_bridge"), p.get("landmark"))
    cx, cy = lms[int(idx)] * (w, h)
    cx += p.get("offset", [0, 0])[0] * face_w
    cy += p.get("offset", [0, 0])[1] * face_w
    sh, sw = bgr.shape[:2]
    x0, y0 = int(cx - sw / 2), int(cy - sh / 2)
    # recorta o adesivo se ele sair da imagem
    ax0, ay0, ax1, ay1 = max(0, -x0), max(0, -y0), sw - max(0, x0 + sw - w), sh - max(0, y0 + sh - h)
    if ax1 <= ax0 or ay1 <= ay0:
        return img, lms
    roi = img[y0 + ay0:y0 + ay1, x0 + ax0:x0 + ax1]
    a = alpha[ay0:ay1, ax0:ax1]
    roi[:] = (bgr[ay0:ay1, ax0:ax1] * a + roi * (1 - a)).astype(np.uint8)
    return img, lms


def _keyframe(keys, t):
    """Interpola linearmente os campos numéricos dos quadros-chave no tempo t."""
    if t <= keys[0]["t"]:
        return keys[0]
    for a, b in zip(keys, keys[1:]):
        if t <= b["t"]:
            w = (t - a["t"]) / max(1e-6, b["t"] - a["t"])
            return {k: a[k] + (b.get(k, a[k]) - a[k]) * w for k in a if isinstance(a[k], (int, float))}
    return keys[-1]


def fx_emoji(img, lms, t, dur, p):
    """Emoji animado na tela (não preso ao rosto).
    keys: [{"t", "x", "y", "size", "rot", "blur", "alpha"}], x/y/size em fração da largura/altura.
    spin: graus/s extras; wobble: amplitude (graus) de balanço. Use emojis de um só caractere
    (sequências com ZWJ, como 🧏‍♀️, não são montadas pela fonte sem a biblioteca raqm)."""
    if _span(t, dur, p) is None:
        return img, lms
    h, w = img.shape[:2]
    kf = _keyframe(p["keys"], t - p.get("at", 0.0))
    px = max(8, int(kf.get("size", 0.25) * w))
    bgr, alpha = _emoji(p["emoji"], px)
    rot = kf.get("rot", 0.0) + p.get("spin", 0.0) * t + p.get("wobble", 0.0) * math.sin(2 * math.pi * 1.5 * t)
    sprite = np.dstack([bgr, (alpha * 255).astype(np.uint8)])
    if rot:
        sh, sw = sprite.shape[:2]
        side = int(math.hypot(sh, sw)) + 2
        canvas = np.zeros((side, side, 4), np.uint8)
        oy, ox = (side - sh) // 2, (side - sw) // 2
        canvas[oy:oy + sh, ox:ox + sw] = sprite
        sprite = cv2.warpAffine(canvas, cv2.getRotationMatrix2D((side / 2, side / 2), rot, 1.0), (side, side))
    blur = int(kf.get("blur", 0) * px)
    if blur > 1:
        sprite = cv2.GaussianBlur(sprite, (0, 0), blur / 3)
    a = sprite[:, :, 3:] / 255.0 * kf.get("alpha", 1.0)
    sh, sw = sprite.shape[:2]
    x0, y0 = int(kf.get("x", 0.5) * w - sw / 2), int(kf.get("y", 0.5) * h - sh / 2)
    ax0, ay0 = max(0, -x0), max(0, -y0)
    ax1, ay1 = sw - max(0, x0 + sw - w), sh - max(0, y0 + sh - h)
    if ax1 > ax0 and ay1 > ay0:
        roi = img[y0 + ay0:y0 + ay1, x0 + ax0:x0 + ax1]
        aa = a[ay0:ay1, ax0:ax1]
        roi[:] = (sprite[ay0:ay1, ax0:ax1, :3] * aa + roi * (1 - aa)).astype(np.uint8)
    return img, lms


EFFECTS = {
    "emoji": fx_emoji, "mirror": fx_mirror,
    "fade_in": fx_fade_in, "fade_out": fx_fade_out, "vignette": fx_vignette, "zoom": fx_zoom, "flash": fx_flash,
    "whip": fx_whip, "shake": fx_shake, "face_sticker": fx_face_sticker,
}


# ---------------------------------------------------------------- player
class TemplatePlayer:
    """Mesma interface do EditPlayer (prepare/load/play/frame/stop)."""

    def __init__(self, audio_bus, replay, archive=None):
        self.bus = audio_bus
        self.replay = replay
        self.archive = archive  # replays de edits anteriores (perpetual_edit/archive.py)
        self.spec = None
        self.t0 = None
        self._caps = {}
        self._seg_clips = {}
        self._archive_thread = None

    def prepare(self, edit_cfg):
        spec = edit_cfg["spec"]
        wav = extract_audio(spec["audio_file"], 0.0, spec["duration"]) if spec.get("audio_file") else None
        self.bus.preload(wav)
        return wav

    def load(self, name, edit_cfg):
        """Chamado no gatilho. O replay só é congelado no play(), depois do carregamento,
        para incluir também o que você fez durante o aviso de "loading"."""
        self.stop()
        self.spec, self.name = edit_cfg["spec"], name
        self.duration = self.spec["duration"]
        self.wav = self.prepare(edit_cfg)
        self.t_trigger = time.perf_counter()
        self._clip, self._times = [], np.zeros(0)
        for i, seg in enumerate(self.spec["segments"]):
            if seg["source"] == "video":
                cap = cv2.VideoCapture(seg["file"])
                cap.set(cv2.CAP_PROP_POS_MSEC, seg.get("from", seg["start"]) * 1000)
                self._caps[i] = [cap, cap.get(cv2.CAP_PROP_FPS) or 30.0, -1, None]
        if any(s["source"] == "keyed" for s in self.spec["segments"]):
            # tela verde: um leitor só, que acompanha o tempo do edit inteiro
            cap = cv2.VideoCapture(self.spec["video"])
            self._caps["keyed"] = [cap, cap.get(cv2.CAP_PROP_FPS) or 30.0, -1, None]
        self._start_archive_load()
        self.t0 = None

    def _start_archive_load(self):
        """Sorteia quais cortes rápidos usarão replays antigos e carrega esses trechos
        do disco numa thread, durante o loading (2 s), para não travar a imagem."""
        self._seg_clips, self._archive_thread = {}, None
        if not (self.archive and self.archive.enabled and self.archive.mix > 0):
            return
        # Só cortes rápidos depois do primeiro trecho: a intro sempre mostra o gesto atual.
        chosen = [i for i, s in enumerate(self.spec["segments"])
                  if i > 0 and s["source"] in ("replay", "keyed") and s["end"] - s["start"] < 1.5
                  and random.random() < self.archive.mix]
        paths = self.archive.pick(2) if chosen else []
        if not paths:
            return
        jobs = []
        for i in chosen:
            seg = self.spec["segments"][i]
            t0 = seg.get("replay_at", -1.5)
            jobs.append((i, random.choice(paths), t0, t0 + (seg["end"] - seg["start"]) * seg.get("speed", 1.0)))

        # Dicionário só deste edit: se a thread atrasar, ela nunca escreve no edit seguinte.
        results = self._seg_clips = {}

        def run():
            metas = {}
            for i, path, a, b in jobs:
                try:
                    if path not in metas:
                        metas[path] = self.archive.load_meta(path)
                    clip = self.archive.load_range(path, a, b, metas[path])
                except Exception as exc:  # replay antigo corrompido: esse corte usa o atual
                    print(f"[template] não consegui ler {path}: {exc}")
                    continue
                if clip:
                    results[i] = (clip, np.array([c[0] for c in clip]))

        self._archive_thread = threading.Thread(target=run, daemon=True)
        self._archive_thread.start()

    def play(self, volume=1.0):
        now = time.perf_counter()
        before = self.spec.get("replay", {}).get("before_s", 1.5)
        self._clip = self.replay.window(self.t_trigger - before, now, self.t_trigger)
        self._times = np.array([c[0] for c in self._clip])
        if self._archive_thread:
            self._archive_thread.join(timeout=0.5)  # normalmente já terminou durante o loading
            if self._seg_clips:
                print(f"[template] {len(self._seg_clips)} corte(s) com replays antigos")
        if self.archive:
            self.archive.save_async(self._clip)  # este replay vira material para os próximos
        self.bus.play(self.wav, volume)
        self.t0 = time.perf_counter()

    @property
    def elapsed(self):
        return 0.0 if self.t0 is None else time.perf_counter() - self.t0

    @property
    def playing(self):
        return self.spec is not None

    def _video_frame(self, i, t_seg):
        cap, fps, idx, cur = self._caps[i]
        target = int(t_seg * fps)
        while idx < target:
            ok, fr = cap.read()
            if not ok:
                break
            idx, cur = idx + 1, fr
        self._caps[i][2:] = [idx, cur]
        return cur

    def _source(self, i, seg, t_seg):
        size = self.spec["size"]
        if seg["source"] == "color":
            color = seg.get("color", [0, 0, 0])  # RGB
            return np.full((size[1], size[0], 3), color[::-1], np.uint8), None
        if seg["source"] == "live":
            _, img, lms = self.replay.latest()
            return fit_aspect(img.copy(), None if lms is None else lms.copy(), size)
        if seg["source"] == "replay":
            return self._replay_frame(seg, t_seg, size, i)
        fr = self._video_frame(i, t_seg)
        img = np.zeros((size[1], size[0], 3), np.uint8) if fr is None else fr
        return fit_aspect(img.copy(), None, size)

    def _replay_frame(self, seg, t_seg, size, i=None):
        """Frame do replay no instante replay_at + t_seg * speed (relativo ao gatilho).
        Usa o replay atual ou, se este corte foi sorteado, um replay antigo do arquivo.
        Em câmera lenta, mistura os dois frames vizinhos para o movimento não "travar"."""
        clip, times = self._seg_clips.get(i, (self._clip, self._times))
        if not clip:
            _, img, lms = self.replay.latest()
            return fit_aspect(img.copy(), lms, size)
        before = self.spec.get("replay", {}).get("before_s", 1.5)
        tau = seg.get("replay_at", -before) + t_seg * seg.get("speed", 1.0)
        tau = float(np.clip(tau, times[0], times[-1]))  # segura a ponta se faltar replay
        j = int(np.searchsorted(times, tau, side="right")) - 1
        j = max(0, min(j, len(clip) - 1))
        _, img, lms = clip[j]
        if seg.get("blend", seg.get("speed", 1.0) < 1.0) and j + 1 < len(clip):
            t_a, t_b = times[j], times[j + 1]
            w = (tau - t_a) / max(1e-6, t_b - t_a)
            img = cv2.addWeighted(img, 1 - w, clip[j + 1][1], w, 0)
            if lms is not None and clip[j + 1][2] is not None:
                lms = lms * (1 - w) + clip[j + 1][2] * w
        else:
            img = img.copy()
        return fit_aspect(img, None if lms is None else lms.copy(), size)

    def frame(self, *_):
        """Frame atual do template, ou None quando terminou."""
        if self.spec is None:
            return None
        t = self.elapsed
        if t >= self.duration:
            self.stop()
            return None
        segs = self.spec["segments"]
        i = next((k for k, s in enumerate(segs) if s["start"] <= t < s["end"]), None)
        if i is None:
            size = self.spec["size"]
            return np.zeros((size[1], size[0], 3), np.uint8)
        seg = segs[i]
        t_seg, dur = t - seg["start"], seg["end"] - seg["start"]
        if seg["source"] == "keyed":
            return self._keyed_frame(i, seg, t, t_seg, dur)
        img, lms = self._source(i, seg, t_seg)
        for fx in seg.get("effects", []):
            img, lms = EFFECTS[fx["type"]](img, lms, t_seg, dur, fx)
        return img

    def _keyed_frame(self, i, seg, t, t_seg, dur):
        """Quadro do vídeo de tela verde no tempo t, com o verde trocado pelo seu replay.
        Os efeitos do trecho (ex.: face_sticker) valem só para a sua imagem."""
        size = self.spec["size"]
        fr = self._video_frame("keyed", t)
        if fr is None:
            return np.zeros((size[1], size[0], 3), np.uint8)
        base = fr if fr.shape[1::-1] == tuple(size) else cv2.resize(fr, tuple(size), interpolation=cv2.INTER_AREA)

        def user_at(box):
            img, lms = self._replay_frame(seg, t_seg, box, i)
            for fx in seg.get("effects", []):
                img, lms = EFFECTS[fx["type"]](img, lms, t_seg, dur, fx)
            return img

        return compose_keyed(base, user_at, self.spec.get("key"))

    def stop(self):
        for cap, *_ in self._caps.values():
            cap.release()
        self._caps = {}
        self._clip = []  # libera a memória do replay congelado
        self._seg_clips = {}
        self.spec, self.t0 = None, None
