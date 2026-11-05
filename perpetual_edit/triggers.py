"""Regras dos gatilhos + robustez (hold de N frames, cooldowns, rearme por borda)."""
import random
import time
from dataclasses import dataclass

import numpy as np


# ---------- avaliadores: (features, cfg_do_gatilho) -> valor medido ----------
# O gatilho fica "ativo" quando valor >= threshold.

def _mouth_open(f, cfg):
    return f.blendshapes.get("jawOpen", 0.0) if f.face_found else 0.0


def _brows_up(f, cfg):
    if not f.face_found:
        return 0.0
    b = f.blendshapes
    return (b.get("browInnerUp", 0.0) + b.get("browOuterUpLeft", 0.0) + b.get("browOuterUpRight", 0.0)) / 3


def _gesture(f, cfg):
    return f.gesture_score if f.gesture == cfg.get("gesture", "Thumb_Up") else 0.0


def _eyes_closed(f, cfg):
    # min: os dois olhos precisam estar fechados (piscar um olho só não conta)
    if not f.face_found:
        return 0.0
    return min(f.blendshapes.get("eyeBlinkLeft", 0.0), f.blendshapes.get("eyeBlinkRight", 0.0))


def _head_turn(f, cfg):
    """Graus de yaw. "side" é relativo à tela: right = nariz para a direita da imagem."""
    if not f.face_found:
        return 0.0
    side = cfg.get("side", "any")
    if side == "left":
        return max(0.0, -f.yaw_deg)
    if side == "right":
        return max(0.0, f.yaw_deg)
    return abs(f.yaw_deg)


def _hand_near_face(f, cfg):
    """1 = mão encostando/na frente do rosto, 0 = a uma largura de rosto ou mais."""
    if not (f.face_found and f.face_bbox and f.hands):
        return 0.0
    a = f.aspect  # converte x para a mesma escala de y
    x0, y0, x1, y1 = f.face_bbox
    face_w = max(1e-3, (x1 - x0) * a)
    best = float("inf")
    for hand in f.hands:
        hx, hy = hand[:, 0] * a, hand[:, 1]
        dx = np.maximum(np.maximum(x0 * a - hx, 0), hx - x1 * a)
        dy = np.maximum(np.maximum(y0 - hy, 0), hy - y1)
        best = min(best, float(np.sqrt(dx * dx + dy * dy).min()))
    return max(0.0, 1.0 - best / face_w)


def _face_lost(f, cfg):
    return f.face_missing_s  # segundos


def _face_back(f, cfg):
    return f.face_back_after_s  # segundos que o rosto ficou fora antes de voltar


def _custom(f, cfg):
    """Probabilidade da classe `to`, mas só logo após o estado estável mudar de `from` para `to`."""
    st = f.custom.get(cfg.get("model"))
    if not st:
        return 0.0
    to, frm = cfg.get("to"), cfg.get("from")
    fresh = (st["stable"] == to and st["since"] <= st["fresh_s"]
             and (frm is None or st["previous"] == frm))
    return st["probs"].get(to, 0.0) if fresh else 0.0


EVALUATORS = {
    "custom": _custom,
    "mouth_open": _mouth_open,
    "brows_up": _brows_up,
    "gesture": _gesture,
    "eyes_closed": _eyes_closed,
    "head_turn": _head_turn,
    "hand_near_face": _hand_near_face,
    "face_lost": _face_lost,
    "face_back": _face_back,
}

# Tipos cujo valor não vem de blendshapes já suavizados: suavizamos aqui.
SMOOTHED_TYPES = {"hand_near_face"}


@dataclass
class TriggerState:
    value: float = 0.0
    active: bool = False
    hold: int = 0
    armed: bool = True
    last_fire: float = -1e9
    last_edit: str | None = None


class TriggerEngine:
    def __init__(self, cfg):
        self.states = {}
        self.muted = set()  # desligados pelo menu (tecla E), sem mexer no config
        self.last_edit_end = -1e9
        self.reconfigure(cfg)

    def reconfigure(self, cfg):
        self.cfg = cfg
        old = self.states
        self.states = {name: old.get(name, TriggerState()) for name in cfg["triggers"]}

    def update(self, features, can_fire, now=None):
        """Atualiza todos os gatilhos; devolve o nome do que disparou (ou None)."""
        now = time.perf_counter() if now is None else now
        fired = None
        global_ready = now - self.last_edit_end >= self.cfg["global_cooldown_s"]
        for name, tcfg in self.cfg["triggers"].items():
            st = self.states[name]
            evaluator = EVALUATORS.get(tcfg["type"])
            if not self.is_on(name) or evaluator is None:
                st.value, st.active, st.hold = 0.0, False, 0
                continue
            value = float(evaluator(features, tcfg))
            if tcfg["type"] in SMOOTHED_TYPES:
                a = self.cfg["smoothing"]
                value = a * value + (1 - a) * st.value
            st.value = value
            st.active = st.value >= tcfg["threshold"]
            st.hold = st.hold + 1 if st.active else 0
            if not st.active:
                st.armed = True  # só rearma depois que a condição some
            ready = (
                can_fire and fired is None and global_ready and st.armed
                and st.hold >= tcfg["hold_frames"]
                and now - st.last_fire >= tcfg["cooldown_s"]
                and tcfg["edits"]
            )
            if ready:
                fired = name
        if fired:
            self.states[fired].last_fire = now
            for st in self.states.values():
                st.armed = False  # a pose que disparou não dispara de novo em seguida
        return fired

    def is_on(self, name):
        return self.cfg["triggers"][name]["enabled"] and name not in self.muted

    def disarm_all(self):
        for st in self.states.values():
            st.armed = False

    def edit_finished(self, now=None):
        self.last_edit_end = time.perf_counter() if now is None else now

    def cooldown_left(self, name, now=None):
        now = time.perf_counter() if now is None else now
        t = self.cfg["triggers"][name]
        own = t["cooldown_s"] - (now - self.states[name].last_fire)
        glob = self.cfg["global_cooldown_s"] - (now - self.last_edit_end)
        return max(0.0, own, glob)

    def choose_edit(self, name):
        """Escolhe um edit do gatilho aleatoriamente, sem repetir o último."""
        st = self.states[name]
        options = self.cfg["triggers"][name]["edits"]
        if len(options) > 1 and st.last_edit in options:
            options = [e for e in options if e != st.last_edit]
        st.last_edit = random.choice(options)
        return st.last_edit
