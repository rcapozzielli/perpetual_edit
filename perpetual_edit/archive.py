"""Arquivo de replays: guarda os replays de edits anteriores para reaproveitar
trechos deles em edits futuros (mais variedade de imagens).

Cada replay vira uma pasta replays/<data>/ com:
  000.jpg, 001.jpg, ...   frames (quadrados, já enquadrados no rosto)
  meta.npz                tempos (relativos ao gatilho) e landmarks
O meta.npz é gravado por último: pasta sem ele = gravação incompleta, ignorada.
Com "clear_on_exit", tudo é apagado ao fechar o app (e ao abrir, caso ele tenha
caído sem fechar direito): a mistura só usa replays da sessão atual.
"""
import os
import random
import re
import shutil
import threading
from datetime import datetime

import cv2
import numpy as np

from .config import resolve

_NAME = re.compile(r"^\d{8}_\d{6}(_\d+)?$")  # só mexemos em pastas criadas por nós


class ReplayArchive:
    def __init__(self, cfg):
        self.configure(cfg)
        self._lock = threading.Lock()
        self._threads = []

    def configure(self, cfg):
        a = cfg.get("archive", {})
        self.enabled = a.get("enabled", True)
        self.dir = resolve(a.get("dir", "replays"))
        self.max_clips = int(a.get("max_clips", 30))
        self.mix = float(a.get("mix", 0.25))
        self.min_face = float(a.get("min_face_ratio", 0.6))
        self.clear_on_exit = bool(a.get("clear_on_exit", True))

    # ---------------------------------------------------------------- gravar
    def save_async(self, clip):
        """Salva o replay numa thread (codificar ~100 JPEGs leva ~0,3 s)."""
        if not self.enabled or not clip:
            return
        with_face = sum(1 for _, _, lms in clip if lms is not None) / len(clip)
        if with_face < self.min_face:
            return  # replay sem rosto (você saiu da câmera): não vale guardar
        th = threading.Thread(target=self._save, args=(list(clip),), daemon=True)
        th.start()
        self._threads = [t for t in self._threads if t.is_alive()] + [th]

    def _save(self, clip):
        with self._lock:
            os.makedirs(self.dir, exist_ok=True)
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
            path, k = os.path.join(self.dir, name), 1
            while os.path.exists(path):
                path, k = os.path.join(self.dir, f"{name}_{k}"), k + 1
            os.makedirs(path)
            times = np.array([t for t, _, _ in clip], np.float32)
            has = np.array([lms is not None for _, _, lms in clip])
            lms = np.stack([l if l is not None else np.zeros((478, 2), np.float32) for _, _, l in clip])
            for i, (_, img, _) in enumerate(clip):
                cv2.imwrite(os.path.join(path, f"{i:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            np.savez_compressed(os.path.join(path, "meta.npz"), times=times, has=has, lms=lms.astype(np.float16))
            self._prune()

    def _prune(self):
        clips = self.list()
        for path in clips[:-self.max_clips] if len(clips) > self.max_clips else []:
            shutil.rmtree(path, ignore_errors=True)

    def clear(self):
        """Apaga todos os replays guardados (espera os que ainda estão sendo salvos).
        Só remove pastas com o nome que nós criamos; qualquer outra coisa na pasta fica."""
        for th in self._threads:
            th.join(timeout=5)
        self._threads = []
        if not os.path.isdir(self.dir):
            return 0
        with self._lock:
            names = [d for d in os.listdir(self.dir) if _NAME.match(d) and os.path.isdir(os.path.join(self.dir, d))]
            for d in names:
                shutil.rmtree(os.path.join(self.dir, d), ignore_errors=True)
            try:
                os.rmdir(self.dir)  # só sai se ficou vazia
            except OSError:
                pass
        return len(names)

    # ---------------------------------------------------------------- ler
    def list(self):
        """Pastas completas, da mais antiga para a mais nova."""
        if not os.path.isdir(self.dir):
            return []
        return [os.path.join(self.dir, d) for d in sorted(os.listdir(self.dir))
                if _NAME.match(d) and os.path.exists(os.path.join(self.dir, d, "meta.npz"))]

    def pick(self, n):
        """Até n replays antigos diferentes, sorteados."""
        clips = self.list()
        return random.sample(clips, min(n, len(clips)))

    @staticmethod
    def load_meta(path):
        meta = np.load(os.path.join(path, "meta.npz"))
        return meta["times"], meta["has"], meta["lms"].astype(np.float32)

    @staticmethod
    def load_range(path, t0, t1, meta=None):
        """Carrega só os frames entre t0 e t1 (tempo relativo ao gatilho daquele replay).
        `meta` (de load_meta) evita reler o meta.npz quando vários cortes usam o mesmo replay."""
        times, has, lms = meta if meta is not None else ReplayArchive.load_meta(path)
        idx = [i for i, t in enumerate(times) if t0 - 0.05 <= t <= t1 + 0.05]
        if not idx:  # pediu fora do que foi gravado: pega o pedaço mais próximo
            j = int(np.argmin(np.abs(times - t0)))
            idx = list(range(max(0, j - 5), min(len(times), j + 15)))
        out = []
        for i in idx:
            img = cv2.imread(os.path.join(path, f"{i:03d}.jpg"))
            if img is not None:
                out.append((float(times[i]), img, lms[i] if has[i] else None))
        return out
