"""HUD estilo vigilância: LIVE, REC, caixa do rosto, painel lateral, rótulo e debug."""
import math
import time
from datetime import datetime

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
MONO = cv2.FONT_HERSHEY_PLAIN
GREEN = (60, 255, 60)
DIM_GREEN = (40, 140, 40)
RED = (40, 40, 255)
WHITE = (235, 235, 235)
GRAY = (130, 130, 130)
YELLOW = (40, 220, 255)
PANEL_W = 330

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def text(img, s, org, scale=0.5, color=WHITE, thick=1, font=FONT, shadow=True):
    x, y = org
    if shadow:
        # Mesma espessura deslocada: espessura maior muda a largura das letras no OpenCV 5.
        cv2.putText(img, s, (x + 2, y + 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), font, scale, color, thick, cv2.LINE_AA)


def _wrap(s, width):
    lines, cur = [], ""
    for word in s.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return lines + [cur] if cur else lines


def _panel_text(img, s, org, scale=0.5, color=WHITE, thick=1, font=FONT):
    text(img, s, org, scale, color, thick, font, shadow=False)  # fundo escuro, sem sombra


def brackets(img, p0, p1, color, length=22, thick=2):
    (x0, y0), (x1, y1) = p0, p1
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thick, cv2.LINE_AA)


def face_cube(bbox, size, yaw_deg, pitch_deg, roll_deg):
    """8 vértices (em px) de um cubo 3D em volta da cabeça, girado com a pose.
    A face da frente fica sobre o rosto e a de trás "fica para trás" quando a cabeça vira.
    Coordenadas da imagem: x para a direita, y para baixo, z saindo da tela."""
    W, H = size
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H
    h = (x1 - x0) * W / 2 * 1.1
    ya, pi, ro = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    ry = np.array([[math.cos(ya), 0, math.sin(ya)], [0, 1, 0], [-math.sin(ya), 0, math.cos(ya)]])
    rx = np.array([[1, 0, 0], [0, math.cos(pi), math.sin(pi)], [0, -math.sin(pi), math.cos(pi)]])
    rz = np.array([[math.cos(ro), -math.sin(ro), 0], [math.sin(ro), math.cos(ro), 0], [0, 0, 1]])
    R = rz @ ry @ rx
    corners = np.array([[sx, sy, sz] for sz in (1, -1) for sy in (-1, 1) for sx in (-1, 1)], float) * h
    pts = corners @ R.T
    f = 6 * h  # perspectiva leve: o que está mais perto fica maior
    k = (f - h) / (f - pts[:, 2])  # normalizado: de frente, a face da frente tem o tamanho do rosto
    xy = pts[:, :2] * k[:, None]
    front = xy[:4].mean(axis=0)  # ancora a face da frente no centro do rosto
    return (xy - front + (cx, cy)).astype(np.int32)


CUBE_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def fmt_hms(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


class HUD:
    def __init__(self):
        self.t_start = time.perf_counter()
        self.label = None
        self.label_until = 0.0
        self.label_dur = 1.5
        self.message = None
        self.message_until = 0.0

    def show_label(self, s, duration=1.5):
        self.label, self.label_dur = s, duration
        self.label_until = time.perf_counter() + duration

    def notify(self, s, duration=3.0):
        print(f"[hud] {s}")
        self.message, self.message_until = s, time.perf_counter() + duration

    # ---------------- sobre a imagem da câmera (vai para a gravação) ----------------
    def draw_overlay(self, img, features, show_face_box, recording_s=None, label_cx=None):
        H, W = img.shape[:2]
        now = time.perf_counter()

        brackets(img, (12, 12), (W - 13, H - 13), DIM_GREEN, 40, 2)

        if int(now * 2) % 2 == 0:
            cv2.circle(img, (38, 40), 9, RED, -1, cv2.LINE_AA)
        text(img, "LIVE", (54, 48), 0.7, WHITE, 2)

        if recording_s is not None:
            rec, color = f"REC {fmt_hms(recording_s)}", RED
        else:
            rec, color = f"REC {fmt_hms(now - self.t_start)}", WHITE
        text(img, rec, (W - 210, 48), 0.7, color, 2)
        text(img, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"), (30, H - 30), 0.5, GREEN)
        text(img, "CAM 01", (W - 100, H - 30), 0.5, GREEN)

        if show_face_box and features.face_found and features.face_bbox:
            # cubo 3D em arame que gira com a cabeça (yaw/pitch do MediaPipe, roll pelos olhos)
            roll = 0.0
            lms = features.face_landmarks
            if lms is not None:
                dx, dy = (lms[263, 0] - lms[33, 0]) * W, (lms[263, 1] - lms[33, 1]) * H
                roll = math.degrees(math.atan2(dy, dx))
            pts = face_cube(features.face_bbox, (W, H), features.yaw_deg, features.pitch_deg, roll)
            for a, b in CUBE_EDGES:
                back = a >= 4 and b >= 4
                cv2.line(img, tuple(pts[a]), tuple(pts[b]), DIM_GREEN if back else GREEN,
                         1 if back else 2, cv2.LINE_AA)
            text(img, "SUBJECT 01", (int(pts[:4, 0].min()), int(pts[:4, 1].min()) - 8), 0.5, GREEN, 1)

        if self.label and now < self.label_until:
            # Rótulo grande que aparece e some (fade nos últimos 30%).
            left = (self.label_until - now) / self.label_dur
            alpha = min(1.0, left / 0.3)
            s = self.label.upper()
            scale, thick = 1.6, 4
            (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
            cx = W // 2 if label_cx is None else label_cx
            x, y = max(30, cx - tw // 2), int(H * 0.2)
            layer = img.copy()
            cv2.rectangle(layer, (x - 20, y - th - 20), (x + tw + 20, y + 20), (0, 0, 0), -1)
            cv2.putText(layer, s, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
            cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, dst=img)
        return img

    def draw_loading(self, img, progress, capturing):
        """Caixa "LOADING EDIT" com barra segmentada, na parte de baixo da imagem."""
        H, W = img.shape[:2]
        progress = max(0.0, min(1.0, progress))
        bw, bh = 460, 96 if capturing else 70
        x0, y0 = (W - bw) // 2, H - bh - 70
        layer = img.copy()
        cv2.rectangle(layer, (x0, y0), (x0 + bw, y0 + bh), (0, 0, 0), -1)
        cv2.addWeighted(layer, 0.7, img, 0.3, 0, dst=img)
        brackets(img, (x0, y0), (x0 + bw, y0 + bh), GREEN, 14, 2)
        text(img, "LOADING EDIT", (x0 + 18, y0 + 28), 0.6, GREEN, 2)
        text(img, f"{int(progress * 100):3d}%", (x0 + bw - 70, y0 + 28), 0.6, GREEN, 2)
        n = 24
        seg_w = (bw - 36) / n
        for i in range(n):
            sx = int(x0 + 18 + i * seg_w)
            filled = (i + 1) / n <= progress + 1e-6
            cv2.rectangle(img, (sx, y0 + 40), (int(sx + seg_w) - 3, y0 + 56), GREEN if filled else (40, 70, 40), -1)
        if capturing:
            if int(time.perf_counter() * 4) % 2 == 0:
                cv2.circle(img, (x0 + 26, y0 + 78), 7, RED, -1, cv2.LINE_AA)
            text(img, "CAPTURANDO REPLAY - SEGURA O GESTO", (x0 + 42, y0 + 84), 0.5, WHITE)
        return img

    # ---------------- debug (só na tela, não grava) ----------------
    def draw_debug(self, img, f):
        H, W = img.shape[:2]
        if f.face_landmarks is not None:
            xs = np.clip((f.face_landmarks[:, 0] * W).astype(int), 0, W - 1)
            ys = np.clip((f.face_landmarks[:, 1] * H).astype(int), 0, H - 1)
            img[ys, xs] = (0, 255, 255)
        for hand in f.hands:
            pts = [(int(x * W), int(y * H)) for x, y, _ in hand]
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, pts[a], pts[b], (255, 200, 0), 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(img, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

        # Caixa com números brutos (bom para calibrar)
        box_w, x, y = 340, 24, 80
        layer = img.copy()
        cv2.rectangle(layer, (x - 10, y - 24), (x + box_w, y + 330), (0, 0, 0), -1)
        cv2.addWeighted(layer, 0.6, img, 0.4, 0, dst=img)
        lines = [
            f"face {'OK' if f.face_found else '--'}  in {f.face_present_s:4.1f}s  out {f.face_missing_s:4.1f}s",
            f"yaw {f.yaw_deg:+6.1f}  pitch {f.pitch_deg:+6.1f}",
            f"gesture {f.gesture} {f.gesture_score:.2f}  hands {len(f.hands)}",
        ]
        for s in lines:
            text(img, s, (x, y), 1.0, GREEN, 1, MONO)
            y += 20
        y += 8
        text(img, "TOP BLENDSHAPES", (x, y), 1.0, GRAY, 1, MONO)
        top = sorted(f.blendshapes.items(), key=lambda kv: kv[1], reverse=True)[:12]
        for name, v in top:
            y += 19
            text(img, f"{name[:18]:<18}", (x, y), 0.9, WHITE, 1, MONO)
            cv2.rectangle(img, (x + 170, y - 9), (x + 170 + int(110 * v), y - 1), GREEN, -1)
        return img

    # ---------------- modo treino ----------------
    def draw_train_overlay(self, img, collector):
        H, W = img.shape[:2]
        x, y, s = 24, 70, 160
        if collector.last_crop is not None:
            img[y:y + s, x:x + s] = cv2.resize(collector.last_crop, (s, s))
            cv2.rectangle(img, (x - 2, y - 2), (x + s + 1, y + s + 1), GREEN, 2)
        else:
            cv2.rectangle(img, (x, y), (x + s, y + s), GRAY, 1)
            text(img, "sem rosto", (x + 30, y + s // 2), 0.5, GRAY)
        text(img, "ROSTO ALINHADO", (x, y + s + 22), 0.45, GREEN)
        if collector.recording:
            s = f"GRAVANDO: {collector.recording}"
            if collector.last_crop is None:
                s += "  (sem rosto, nada salvo)"
            (tw, _), _ = cv2.getTextSize(s, FONT, 1.0, 2)
            cx = (W - tw) // 2
            if int(time.perf_counter() * 2) % 2 == 0:
                cv2.circle(img, (cx - 24, H - 72), 11, RED, -1, cv2.LINE_AA)
            text(img, s, (cx, H - 62), 1.0, RED, 2)
        return img

    def draw_prompt(self, img, label, value):
        """Campo de texto no centro da imagem (criar modelo no modo treino)."""
        H, W = img.shape[:2]
        bw, bh = min(W - 80, 720), 150
        x0, y0 = (W - bw) // 2, (H - bh) // 2
        layer = img.copy()
        cv2.rectangle(layer, (x0, y0), (x0 + bw, y0 + bh), (0, 0, 0), -1)
        cv2.addWeighted(layer, 0.85, img, 0.15, 0, dst=img)
        brackets(img, (x0, y0), (x0 + bw, y0 + bh), YELLOW, 16, 2)
        text(img, label, (x0 + 20, y0 + 34), 0.6, YELLOW, 1)
        cv2.rectangle(img, (x0 + 20, y0 + 52), (x0 + bw - 20, y0 + 96), (60, 60, 60), 1)
        cursor = "_" if int(time.perf_counter() * 2) % 2 == 0 else " "
        text(img, value[-40:] + cursor, (x0 + 32, y0 + 84), 0.8, WHITE, 2)
        text(img, "ENTER confirma   ESC cancela   (sem acentos)", (x0 + 20, y0 + 128), 0.45, GRAY)
        return img

    def draw_train_panel(self, height, collector, live, fps):
        p = np.full((height, PANEL_W, 3), 14, np.uint8)
        text = _panel_text
        cv2.line(p, (0, 0), (0, height), YELLOW, 2)
        y = 34
        text(p, "MODO TREINO", (16, y), 0.65, YELLOW, 2)
        y += 22
        text(p, f"{fps:4.1f} FPS", (16, y), 1.0, GRAY, 1, MONO)
        if not collector.model:
            y += 40
            for s in ("nenhum modelo em", "\"custom_models\" no config.json", "(veja o README)"):
                text(p, s, (16, y), 0.45, WHITE)
                y += 22
        else:
            y += 40
            text(p, "MODELO", (16, y), 0.45, GRAY)
            text(p, f"{collector.idx + 1}/{len(collector.names)}  TAB troca", (PANEL_W - 140, y), 0.9, GRAY, 1, MONO)
            y += 28
            text(p, collector.model, (16, y), 0.7, WHITE, 2)
            y += 34
            text(p, "CLASSES  (amostras, sessões / prob.)", (16, y), 0.42, GRAY)
            counts, sessions = collector.counts(), collector.sessions()
            for i, cls in enumerate(collector.classes):
                y += 34
                rec = collector.recording == cls
                color = RED if rec else WHITE
                text(p, f"[{i + 1}] {cls}", (16, y), 0.5, color, 2 if rec else 1)
                n_ses = sessions.get(cls, 0)
                text(p, f"{counts.get(cls, 0)}", (PANEL_W - 104, y), 0.5, color)
                text(p, f"{n_ses}s", (PANEL_W - 44, y), 0.5, YELLOW if n_ses < 2 else color)
                if live:
                    pv = live["probs"].get(cls, 0.0)
                    by = y + 8
                    cv2.rectangle(p, (16, by), (PANEL_W - 16, by + 5), (50, 50, 50), -1)
                    cv2.rectangle(p, (16, by), (16 + int((PANEL_W - 32) * pv), by + 5),
                                  GREEN if live["stable"] == cls else GRAY, -1)
            y += 36
            if live:
                text(p, f"estado estável: {live['stable'] or '-'}", (16, y), 0.45, GREEN)
            else:
                text(p, "modelo ainda não treinado", (16, y), 0.45, GRAY)

        if self.message and time.perf_counter() < self.message_until:
            y = height - 160
            for line in _wrap(self.message, 34):
                text(p, line, (16, y), 0.42, YELLOW)
                y += 18
        help_lines = ["1-9 grava/para   N novo modelo", "ENTER treina   TAB modelo",
                      "U desfaz sessão  X apaga (0 = modelo)", "T sai do treino   Q sair"]
        for i, s in enumerate(help_lines):
            text(p, s, (16, height - 80 + i * 20), 1.0, GRAY, 1, MONO)
        return p

    # ---------------- painel lateral (só na tela, não grava) ----------------
    def draw_panel(self, height, state, engine, cfg, fps, debug):
        p = np.full((height, PANEL_W, 3), 14, np.uint8)
        text = _panel_text
        cv2.line(p, (0, 0), (0, height), DIM_GREEN, 2)
        now = time.perf_counter()
        y = 34
        text(p, "PERPETUAL_EDIT", (16, y), 0.65, GREEN, 2)
        y += 22
        text(p, f"MONITOR v1   {fps:4.1f} FPS   {cfg['overlay_mode'].upper()}", (16, y), 1.0, GRAY, 1, MONO)

        y += 40
        text(p, "STATUS", (16, y), 0.45, GRAY)
        y += 30
        status_color = {"WAITING": GREEN, "PRE_EDIT": YELLOW, "PLAYING": RED}.get(state["mode"], WHITE)
        text(p, state["status"], (16, y), 0.7, status_color, 2)
        if state.get("detail"):
            y += 22
            text(p, state["detail"], (16, y), 0.45, WHITE)

        y += 36
        text(p, "TRIGGERS", (16, y), 0.45, GRAY)
        y += 8
        for name, tcfg in cfg["triggers"].items():
            st = engine.states[name]
            y += 34
            if y > height - 110:
                break
            enabled = engine.is_on(name)
            cd = engine.cooldown_left(name, now)
            tag = "OFF" if not enabled else (f"{cd:3.1f}s" if cd > 0 else "READY")
            color = GRAY if not enabled else (GREEN if st.active else WHITE)
            text(p, name, (16, y), 0.48, color)
            text(p, tag, (PANEL_W - 58, y), 0.38, GRAY if cd > 0 or not enabled else GREEN)
            # Barra valor x limiar. Valores 0-1 usam a escala direta; graus/segundos vão até 2x o limiar.
            thr = tcfg["threshold"]
            vmax = 1.0 if thr <= 1 else thr * 2
            bx0, bx1, by = 16, PANEL_W - 16, y + 8
            cv2.rectangle(p, (bx0, by), (bx1, by + 6), (50, 50, 50), -1)
            frac = max(0.0, min(1.0, st.value / vmax))
            cv2.rectangle(p, (bx0, by), (bx0 + int((bx1 - bx0) * frac), by + 6), color, -1)
            tx = bx0 + int((bx1 - bx0) * min(1.0, thr / vmax))
            cv2.line(p, (tx, by - 3), (tx, by + 9), YELLOW, 2)
            if debug:
                fmt = "{:.2f}" if thr <= 1 else "{:.1f}"
                text(p, f"{fmt.format(st.value)}/{fmt.format(thr)} h{min(st.hold, 99)}/{tcfg['hold_frames']}",
                     (138, y), 0.8, GRAY, 1, MONO)

        if self.message and now < self.message_until:
            text(p, self.message[:40], (16, height - 90), 0.42, YELLOW)
        text(p, "Q sair  D debug  R recarregar", (16, height - 60), 1.0, GRAY, 1, MONO)
        text(p, "ESPAÇO edit  G gravar  T treino", (16, height - 40), 1.0, GRAY, 1, MONO)
        text(p, "E gatilhos  H esconde painel", (16, height - 20), 1.0, GRAY, 1, MONO)
        return p

    def draw_trigger_menu(self, height, cfg, engine, cursor):
        """Painel do menu de gatilhos (tecla E): liga/desliga cada um."""
        p = np.full((height, PANEL_W, 3), 14, np.uint8)
        text = _panel_text
        cv2.line(p, (0, 0), (0, height), YELLOW, 2)
        y = 34
        text(p, "GATILHOS", (16, y), 0.65, YELLOW, 2)
        y += 22
        on = sum(engine.is_on(n) for n in cfg["triggers"])
        text(p, f"{on}/{len(cfg['triggers'])} ligados", (16, y), 1.0, GRAY, 1, MONO)
        y += 16
        row = 30
        names = list(cfg["triggers"])
        visible = max(1, (height - y - 110) // row)
        first = min(max(0, cursor - visible // 2), max(0, len(names) - visible))  # rola com o cursor
        for i in range(first, min(len(names), first + visible)):
            name = names[i]
            y += row
            locked = not cfg["triggers"][name]["enabled"]  # desligado no config: o menu não liga
            is_on = engine.is_on(name)
            if i == cursor:
                cv2.rectangle(p, (8, y - 21), (PANEL_W - 8, y + 9), (60, 60, 60), -1)
                cv2.rectangle(p, (8, y - 21), (PANEL_W - 8, y + 9), YELLOW, 1)
            box = (20, y - 14)
            cv2.rectangle(p, box, (box[0] + 16, box[1] + 16), GREEN if is_on else GRAY, 1)
            if is_on:
                cv2.line(p, (box[0] + 3, box[1] + 8), (box[0] + 7, box[1] + 12), GREEN, 2, cv2.LINE_AA)
                cv2.line(p, (box[0] + 7, box[1] + 12), (box[0] + 13, box[1] + 3), GREEN, 2, cv2.LINE_AA)
            text(p, name, (48, y), 0.5, WHITE if is_on else GRAY)
            if locked:
                text(p, "config", (PANEL_W - 70, y), 0.38, GRAY)
        if self.message and time.perf_counter() < self.message_until:
            text(p, self.message[:40], (16, height - 90), 0.42, YELLOW)
        help_lines = ["W/S ou setas: mover", "ESPAÇO liga/desliga  A todos", "E ou ESC fecha"]
        for i, s in enumerate(help_lines):
            text(p, s, (16, height - 60 + i * 20), 1.0, GRAY, 1, MONO)
        return p

    def draw_toast(self, img):
        """Avisos na própria imagem quando o painel está escondido (não vai para a gravação)."""
        if self.message and time.perf_counter() < self.message_until:
            H = img.shape[0]
            text(img, self.message[:80], (30, H - 60), 0.55, YELLOW)
