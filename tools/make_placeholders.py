"""Gera clipes e sons de teste para usar sem ter edits reais.

Cada clipe tem um "flash" branco + bipe a cada segundo inteiro: se o flash
e o bipe coincidirem no app, a sincronia áudio/vídeo está ok.

Uso:  python tools/make_placeholders.py
"""
import os
import sys
import wave

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perpetual_edit.audio import run_ffmpeg  # noqa: E402
from perpetual_edit.config import ROOT  # noqa: E402

SR = 44100
FPS = 30

CLIPS = [
    # nome, (w, h), duração, cor BGR, frequência base
    ("edit_1", (540, 960), 4.5, (180, 40, 200), 440),   # vertical 9:16
    ("edit_2", (1280, 720), 3.0, (200, 160, 30), 554),  # horizontal 16:9
    ("edit_3", (720, 720), 3.5, (30, 120, 240), 660),   # quadrado
]


def write_wav(path, samples):
    samples = np.clip(samples, -1, 1)
    stereo = np.repeat((samples * 32767).astype(np.int16)[:, None], 2, axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())


def tone(freq, dur, vol=0.5):
    t = np.arange(int(SR * dur)) / SR
    env = np.minimum(1, np.minimum(t / 0.01, (dur - t) / 0.03).clip(0))
    return vol * env * np.sin(2 * np.pi * freq * t)


def loading_sound(dur):
    """Tique-taques que aceleram + "riser" subindo: avisa que o edit está carregando."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    out = np.zeros(n)
    # riser: seno com frequência subindo e volume crescendo
    freq = 180 + 900 * (t / dur) ** 2
    out += 0.18 * (t / dur) ** 1.5 * np.sin(2 * np.pi * np.cumsum(freq) / SR)
    # ticks: o intervalo cai de 0,25 s para 0,06 s
    pos = 0.0
    while pos < dur - 0.05:
        k = pos / dur
        tick = tone(1500 + 800 * k, 0.025, 0.35 + 0.25 * k)
        i = int(pos * SR)
        out[i:i + len(tick)] += tick[: n - i]
        pos += 0.25 - 0.19 * k
    return out


def clip_audio(dur, base):
    t = np.arange(int(SR * dur)) / SR
    pad = 0.12 * np.sin(2 * np.pi * base / 2 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))
    for s in range(int(dur) + 1):
        b = tone(base * 2, 0.12, 0.6)
        i = int(s * SR)
        pad[i:i + len(b)] += b[: max(0, len(pad) - i)]
    return pad


def clip_video(path, name, size, dur, color):
    w, h = size
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    n = int(dur * FPS)
    for i in range(n):
        t = i / FPS
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = color
        # faixas diagonais animadas
        for k in range(-h, w, 80):
            x = int(k + (t * 200) % 80)
            cv2.line(img, (x, 0), (x + h, h), tuple(int(c * 0.7) for c in color), 18)
        if t - int(t) < 0.1:
            img[:] = (255, 255, 255)
        s = name.upper()
        (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, 2.2, 5)
        cv2.putText(img, s, ((w - tw) // 2, h // 2), cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 0, 0), 5, cv2.LINE_AA)
        ts = f"{t:4.2f}s"
        (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(img, ts, ((w - tw) // 2, h // 2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.rectangle(img, (0, h - 24), (int(w * (i + 1) / n), h), (255, 255, 255), -1)
        writer.write(img)
    writer.release()


def main():
    vdir, sdir, tmp = (os.path.join(ROOT, d) for d in ("videos", "sounds", "cache"))
    for d in (vdir, sdir, tmp):
        os.makedirs(d, exist_ok=True)

    for name, size, dur, color, base in CLIPS:
        out = os.path.join(vdir, f"{name}.mp4")
        if os.path.exists(out):
            print(f"já existe, pulando: {out}")
            continue
        v, a = os.path.join(tmp, f"_{name}.mp4"), os.path.join(tmp, f"_{name}.wav")
        clip_video(v, name, size, dur, color)
        write_wav(a, clip_audio(dur, base))
        res = run_ffmpeg(["-i", v, "-i", a, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                          "-c:a", "aac", "-b:a", "160k", "-shortest", out])
        os.remove(v)
        os.remove(a)
        if res.returncode != 0:
            raise SystemExit(f"ffmpeg falhou em {name}: {res.stderr}")
        print(f"ok: {out}")

    sounds = {
        "pre.wav": np.concatenate([
            # "whoosh" ascendente + dois bipes
            0.35 * np.sin(2 * np.pi * np.cumsum(np.linspace(300, 1600, int(SR * 0.35))) / SR)
            * np.linspace(0.2, 1, int(SR * 0.35)),
            tone(1760, 0.08, 0.5), np.zeros(int(SR * 0.04)), tone(2093, 0.12, 0.5),
        ]),
        "post.wav": np.concatenate([tone(880, 0.08, 0.4), tone(587, 0.14, 0.4)]),
        "loading.wav": loading_sound(2.0),
    }
    for fname, samples in sounds.items():
        out = os.path.join(sdir, fname)
        if os.path.exists(out):
            print(f"já existe, pulando: {out}")
            continue
        write_wav(out, samples)
        print(f"ok: {out}")


if __name__ == "__main__":
    main()
