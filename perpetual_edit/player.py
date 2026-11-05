"""Player de edits: frames do clipe via OpenCV, áudio via pygame.

Sincronia: o áudio começa a tocar e, no mesmo instante, guardamos t0.
A cada frame da câmera mostramos o frame do clipe correspondente a
(agora - t0) * fps_do_clipe, pulando ou repetindo frames. O áudio nunca
espera pelo vídeo; é o vídeo que acompanha o relógio do áudio.
"""
import math
import random
import time

import cv2
import numpy as np

from .audio import extract_audio


class EditPlayer:
    def __init__(self, audio_bus):
        self.bus = audio_bus
        self.cap = None
        self.name = None
        self.t0 = None

    def prepare(self, edit_cfg):
        """Aquece o cache de áudio (chamado no início e ao recarregar o config)."""
        wav = extract_audio(edit_cfg["file"], edit_cfg["start"], edit_cfg["end"])
        self.bus.preload(wav)
        return wav

    def load(self, name, edit_cfg):
        """Abre o clipe e posiciona no início. Feito durante o som de pré-edit
        para esconder o tempo de abertura."""
        self.stop()
        cap = cv2.VideoCapture(edit_cfg["file"])
        if not cap.isOpened():
            raise RuntimeError(f"Não consegui abrir {edit_cfg['file']}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        start = edit_cfg["start"]
        end = edit_cfg["end"] if edit_cfg["end"] is not None else (n_frames / fps if n_frames else 1e9)
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        ok, first = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Clipe vazio ou trecho fora do vídeo: {edit_cfg['file']}")
        self.cap, self.name, self.fps = cap, name, fps
        self.duration = max(0.1, end - start)
        self.current, self.idx = first, 0
        self.wav = self.prepare(edit_cfg)
        self.t0 = None

    def play(self, volume=1.0):
        self.channel = self.bus.play(self.wav, volume)
        self.t0 = time.perf_counter()

    @property
    def elapsed(self):
        return 0.0 if self.t0 is None else time.perf_counter() - self.t0

    def frame(self):
        """Frame atual do clipe, ou None quando o clipe terminou."""
        if self.cap is None:
            return None
        if self.t0 is None:
            return self.current
        t = self.elapsed
        if t >= self.duration:
            self.stop()
            return None
        target = int(t * self.fps)
        if target > self.idx:
            for _ in range(target - self.idx - 1):
                self.cap.grab()  # pula sem converter a imagem
            ok, fr = self.cap.read()
            if not ok:
                self.stop()
                return None
            self.current, self.idx = fr, target
        return self.current

    def stop(self):
        if self.cap is not None:
            self.cap.release()
        self.cap, self.t0 = None, None

    @property
    def playing(self):
        return self.cap is not None


def _fit(img, w, h):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    return cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_LINEAR)


def composite(cam, clip, mode="fullscreen", pip=None):
    """Coloca o clipe por cima da câmera (tela cheia com fundo desfocado, ou PiP)."""
    H, W = cam.shape[:2]
    if mode == "pip":
        pip = pip or {}
        fg = _fit(clip, int(W * pip.get("scale", 0.38)), int(H * pip.get("scale", 0.38) * 1.6))
        fh, fw = fg.shape[:2]
        m = pip.get("margin", 24)
        corner = pip.get("corner", "bottom_right")
        x = m if "left" in corner else W - fw - m
        y = m if "top" in corner else H - fh - m
        out = cam.copy()
        out[y:y + fh, x:x + fw] = fg
        cv2.rectangle(out, (x - 2, y - 2), (x + fw + 1, y + fh + 1), (0, 255, 0), 2)
        return out

    fg = _fit(clip, W, H)
    fh, fw = fg.shape[:2]
    if fw >= W - 2 and fh >= H - 2:
        return cv2.resize(fg, (W, H))
    # Fundo: o próprio clipe ampliado, desfocado e escurecido (estilo TikTok).
    ih, iw = clip.shape[:2]
    s = max(W / iw, H / ih)
    small = cv2.resize(clip, (max(1, int(iw * s / 8)), max(1, int(ih * s / 8))))
    small = cv2.GaussianBlur(small, (0, 0), 3)
    bg = cv2.resize(small, (max(W, int(iw * s)), max(H, int(ih * s))))
    by, bx = (bg.shape[0] - H) // 2, (bg.shape[1] - W) // 2
    out = cv2.convertScaleAbs(bg[by:by + H, bx:bx + W], alpha=0.45)
    y, x = (H - fh) // 2, (W - fw) // 2
    out[y:y + fh, x:x + fw] = fg
    return out


def _ease_out_back(x, k=1.9):
    """0 -> 1 passando um pouco de 1 (o "quique" do pop)."""
    x -= 1
    return 1 + (k + 1) * x ** 3 + k * x ** 2


class Popup:
    """O edit como um card que "pula" na tela, em posição sorteada longe do rosto."""

    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.rect = None  # (x, y, w, h) do card em tamanho final
        self.angle = 0.0

    def start(self, frame_shape, clip_aspect, face_bbox=None, rng=random):
        """Sorteia posição e inclinação. clip_aspect = largura/altura do edit."""
        H, W = frame_shape[:2]
        c = self.cfg
        h = int(H * c.get("height", 0.55))
        w = int(h * clip_aspect)
        if w > W * 0.8:  # edit muito largo: limita pela largura
            w = int(W * 0.8)
            h = int(w / clip_aspect)
        m = int(c.get("margin", 30))
        face = None
        if face_bbox:
            x0, y0, x1, y1 = face_bbox
            pad = 0.05
            face = ((x0 - pad) * W, (y0 - pad) * H, (x1 + pad) * W, (y1 + pad) * H)
        candidates = []
        for _ in range(40):
            x = rng.randint(m, max(m, W - w - m))
            y = rng.randint(m, max(m, H - h - m))
            overlap = 0.0
            if face:
                ox = max(0.0, min(x + w, face[2]) - max(x, face[0]))
                oy = max(0.0, min(y + h, face[3]) - max(y, face[1]))
                overlap = ox * oy
            candidates.append((overlap, x, y))
        best = min(o for o, _, _ in candidates)
        # entre as posições que cobrem menos o rosto, qualquer uma serve (variedade)
        good = [(x, y) for o, x, y in candidates if o <= best + 0.05 * w * h]
        x, y = rng.choice(good)
        self.rect = (x, y, w, h)
        tilt = float(c.get("tilt_deg", 4))
        self.angle = rng.uniform(-tilt, tilt)

    def draw(self, cam, clip, elapsed, remaining):
        """Desenha o card sobre uma cópia da câmera. elapsed/remaining em segundos."""
        if self.rect is None:
            self.start(cam.shape, clip.shape[1] / clip.shape[0])
        x, y, w, h = self.rect
        t_in, t_out = 0.28, 0.18
        if remaining < t_out:
            scale = max(0.0, remaining / t_out) ** 2
        else:
            scale = _ease_out_back(min(1.0, elapsed / t_in))
        out = cam.copy()
        if scale < 0.03:
            return out
        b = int(self.cfg.get("border", 6))
        card = np.full((h + 2 * b, w + 2 * b, 3), 255, np.uint8)
        fitted = cv2.resize(clip, (w, h), interpolation=cv2.INTER_LINEAR)
        card[b:b + h, b:b + w] = fitted
        ch, cw = card.shape[:2]
        cx, cy = x + w / 2, y + h / 2
        # card (cw x ch) -> tela: escala, gira em volta do centro e leva até (cx, cy)
        a = math.radians(self.angle)
        ca, sa = math.cos(a) * scale, math.sin(a) * scale
        M = np.array([[ca, -sa, cx - ca * cw / 2 + sa * ch / 2],
                      [sa, ca, cy - sa * cw / 2 - ca * ch / 2]], np.float32)
        corners = np.array([[0, 0, 1], [cw, 0, 1], [cw, ch, 1], [0, ch, 1]], np.float32) @ M.T
        # sombra: o mesmo contorno deslocado, escurecendo a câmera por baixo
        off = max(4, int(12 * scale))
        shadow = out.copy()
        cv2.fillConvexPoly(shadow, (corners + off).astype(np.int32), (0, 0, 0), cv2.LINE_AA)
        cv2.addWeighted(shadow, 0.45, out, 0.55, 0, dst=out)
        # BORDER_TRANSPARENT: só os pixels cobertos pelo card são escritos
        cv2.warpAffine(card, M, (out.shape[1], out.shape[0]), dst=out,
                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        return out


class Slot:
    """Janela fixa no canto da câmera (estilo monitor de vigilância): mostra
    "WAITING...", vira "EDITING..." no loading e o edit toca dentro dela."""

    GREEN = (60, 255, 60)
    DIM = (40, 150, 40)

    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.big = False  # tamanho do edit atual (sorteado em start)
        self.t_grow = 0.0

    def start(self, rng=random):
        c = self.cfg
        self.big = rng.random() < float(c.get("big_chance", 0.33))
        self.t_grow = time.perf_counter()

    def rect(self, frame_shape, big=False):
        H, W = frame_shape[:2]
        c = self.cfg
        aspect = float(c.get("aspect", 1.4))
        w = int(W * float(c.get("big_width" if big else "width", 0.55 if big else 0.28)))
        h = int(w / aspect)
        mx, top = int(c.get("margin", 40)), int(c.get("top", 72))
        h = min(h, H - top - mx)
        x = mx if "left" in c.get("corner", "top_right") else W - w - mx
        return x, top, w, h

    def _current_rect(self, frame_shape, active):
        """No gatilho, a janela cresce suavemente até o tamanho do edit (se for grande)."""
        small = self.rect(frame_shape, False)
        if not (active and self.big):
            return small
        big = self.rect(frame_shape, True)
        k = min(1.0, (time.perf_counter() - self.t_grow) / 0.35)
        k = 1 - (1 - k) ** 3
        return tuple(int(a + (b - a) * k) for a, b in zip(small, big))

    @staticmethod
    def _text(img, s, org, scale, color, thick=1):
        cv2.putText(img, s, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def draw_idle(self, img, loading=None):
        """Janela sem edit. loading=None -> WAITING; 0..1 -> EDITING (progresso)."""
        x, y, w, h = self._current_rect(img.shape, loading is not None)
        now = time.perf_counter()
        roi = img[y:y + h, x:x + w]
        roi[:] = cv2.convertScaleAbs(roi, alpha=0.25)  # fundo escurecido
        active = loading is not None
        color = self.GREEN if active else self.DIM
        cx, cy = x + w // 2, y + h // 2
        r = int(min(w, h) * 0.2)
        # mira: hexágono + cruz
        hexagon = np.array([[cx + r * math.cos(math.radians(60 * i + 30)), cy + r * math.sin(math.radians(60 * i + 30))]
                            for i in range(6)], np.int32)
        cv2.polylines(img, [hexagon], True, color, 1, cv2.LINE_AA)
        cv2.line(img, (cx - int(r * 1.7), cy), (cx + int(r * 1.7), cy), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - int(r * 1.5)), (cx, cy + int(r * 1.5)), color, 1, cv2.LINE_AA)
        dots = "." * (1 + int(now * 3) % 3)
        label = ("EDITING" if active else "WAITING") + dots
        scale = max(0.42, min(w, h) / 380)
        (tw, th), _ = cv2.getTextSize("WAITING...", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        tx, ty = cx - tw // 2, cy + th // 2
        img[ty - th - 4:ty + 5, tx - 6:tx + tw + 6] //= 3  # fundo atrás do texto, para ler por cima da mira
        self._text(img, label, (tx, ty), scale, (230, 255, 230) if active else self.GREEN)
        if active:
            # pontinhos girando em volta da mira (o "carregando")
            for i in range(6):
                a = now * 3.2 + i * math.pi / 3
                rr = r * (1.35 + 0.15 * math.sin(now * 5 + i))
                p = (int(cx + rr * math.cos(a)), int(cy + rr * math.sin(a)))
                cv2.circle(img, p, 2, self.GREEN, -1, cv2.LINE_AA)
            # barrinha de progresso discreta na base da janela
            bw = int((w - 16) * max(0.0, min(1.0, loading)))
            cv2.rectangle(img, (x + 8, y + h - 10), (x + 8 + bw, y + h - 7), self.GREEN, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2 if active else 1, cv2.LINE_AA)
        return img

    def draw_clip(self, cam, clip):
        """O edit dentro da janela (corte seco, sem animação)."""
        out = cam.copy()
        x, y, w, h = self._current_rect(cam.shape, True)
        ch, cw = clip.shape[:2]
        if max(cw / ch, (w / h)) / min(cw / ch, (w / h)) < 1.6:
            # proporções parecidas: corta as sobras e preenche a janela
            s = max(w / cw, h / ch)
            big = cv2.resize(clip, (max(w, int(cw * s)), max(h, int(ch * s))), interpolation=cv2.INTER_LINEAR)
            oy, ox = (big.shape[0] - h) // 2, (big.shape[1] - w) // 2
            out[y:y + h, x:x + w] = big[oy:oy + h, ox:ox + w]
        else:
            # muito diferente (ex.: 9:16 numa janela deitada): inteiro, com fundo desfocado
            out[y:y + h, x:x + w] = composite(np.zeros((h, w, 3), np.uint8), clip)
        cv2.rectangle(out, (x, y), (x + w, y + h), self.GREEN, 2, cv2.LINE_AA)
        return out
