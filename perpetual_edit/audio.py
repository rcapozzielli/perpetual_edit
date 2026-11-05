"""Áudio: mixer do pygame, extração/cache do áudio dos clipes e um "barramento"
que avisa quem estiver ouvindo (o gravador de sessão) sobre cada som tocado."""
import hashlib
import os
import subprocess
import sys
import time

import imageio_ffmpeg

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from .config import ROOT

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
CACHE_DIR = os.path.join(ROOT, "cache")
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def run_ffmpeg(args):
    return subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *args],
                          capture_output=True, text=True, creationflags=_NO_WINDOW)


def extract_audio(src, start=0.0, end=None):
    """Extrai o trecho [start, end] do áudio de `src` para um .wav em cache/.
    Devolve o caminho do .wav, ou None se o arquivo não tiver áudio."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    st = os.stat(src)
    key = hashlib.sha1(f"{os.path.abspath(src)}|{st.st_mtime}|{st.st_size}|{start}|{end}".encode()).hexdigest()[:10]
    base = os.path.join(CACHE_DIR, f"{os.path.splitext(os.path.basename(src))[0]}_{key}")
    wav, no_audio = base + ".wav", base + ".noaudio"
    if os.path.exists(wav):
        return wav
    if os.path.exists(no_audio):
        return None
    args = ["-ss", f"{start:.3f}", "-i", src]
    if end is not None:
        args += ["-t", f"{end - start:.3f}"]
    args += ["-vn", "-ac", "2", "-ar", "44100", "-acodec", "pcm_s16le", wav]
    res = run_ffmpeg(args)
    if res.returncode != 0 or not os.path.exists(wav) or os.path.getsize(wav) <= 44:
        if os.path.exists(wav):
            os.remove(wav)
        open(no_audio, "w").close()
        return None
    return wav


class AudioBus:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(16)
        self._sounds = {}
        self.listeners = []  # callables (path, t_perf, volume)

    def _sound(self, path):
        if path not in self._sounds:
            self._sounds[path] = pygame.mixer.Sound(path)
        return self._sounds[path]

    def preload(self, path):
        if path:
            self._sound(path)

    def play(self, path, volume=1.0):
        if not path:
            return None
        snd = self._sound(path)
        snd.set_volume(volume)
        channel = snd.play()
        t = time.perf_counter()
        for cb in self.listeners:
            cb(path, t, volume)
        return channel

    def clear_cache(self):
        self._sounds.clear()

    def stop_all(self):
        pygame.mixer.stop()

    def close(self):
        pygame.mixer.quit()
