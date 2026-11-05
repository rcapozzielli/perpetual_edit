"""Gravação da sessão (tecla G): câmera + edits + sons, num mp4 pronto para postar.

Vídeo: os frames compostos vão para um mp4 temporário a 30 fps fixos. Para o
tempo do vídeo bater com o relógio, cada frame é repetido ou descartado
conforme o tempo real (a câmera não roda a exatamente 30 fps).

Áudio: não gravamos o áudio do sistema; anotamos cada som tocado (arquivo +
instante). Ao parar, o ffmpeg posiciona cada som no instante certo (adelay),
mistura tudo (amix) e junta com o vídeo em H.264 + AAC.
"""
import os
import queue
import threading
import time
from datetime import datetime

import cv2

from .audio import run_ffmpeg
from .config import ROOT

REC_DIR = os.path.join(ROOT, "recordings")
FPS = 30


class SessionRecorder:
    def __init__(self, audio_bus, on_done=None):
        self.bus = audio_bus
        self.on_done = on_done or print
        self.writer = None
        self._threads = []
        audio_bus.listeners.append(self._on_sound)

    @property
    def active(self):
        return self.writer is not None

    @property
    def elapsed(self):
        return time.perf_counter() - self.t0 if self.active else 0.0

    def start(self, width, height):
        os.makedirs(REC_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_path = os.path.join(REC_DIR, f"sessao_{stamp}.mp4")
        self.tmp_path = os.path.join(REC_DIR, f"_tmp_{stamp}.mp4")
        self.writer = cv2.VideoWriter(self.tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
        self.size = (width, height)
        self.t0 = time.perf_counter()
        self.n_written = 0
        self.sounds = []
        self._queue = queue.Queue(maxsize=90)  # ~3 s de folga
        self._writer_thread = threading.Thread(target=self._write_loop, args=(self.writer, self._queue), daemon=True)
        self._writer_thread.start()

    def _on_sound(self, path, t, volume):
        if self.active and path:
            self.sounds.append((path, max(0.0, t - self.t0), volume))

    def write(self, frame):
        """Enfileira o frame; a codificação acontece na thread de escrita."""
        if not self.active:
            return
        target = int(self.elapsed * FPS) + 1
        repeat = target - self.n_written  # >1 repete o frame se a câmera atrasou
        if repeat > 0:
            self.n_written = target
            self._queue.put((frame.copy(), repeat))

    def _write_loop(self, writer, q):
        while True:
            item = q.get()
            if item is None:
                break
            frame, repeat = item
            if frame.shape[1::-1] != self.size:
                frame = cv2.resize(frame, self.size)
            for _ in range(repeat):
                writer.write(frame)
        writer.release()

    def stop(self):
        """Fecha o vídeo e faz a mixagem final numa thread (não trava a janela)."""
        if not self.active:
            return
        self._queue.put(None)
        self._writer_thread.join()
        self.writer = None
        job = (self.tmp_path, self.out_path, self.n_written / FPS, list(self.sounds))
        th = threading.Thread(target=self._finalize, args=job, daemon=True)
        th.start()
        self._threads.append(th)

    def _finalize(self, tmp, out, duration, sounds):
        args = ["-i", tmp]
        for path, _, _ in sounds:
            args += ["-i", path]
        if sounds:
            chains, labels = [], []
            for i, (_, t, vol) in enumerate(sounds, start=1):
                ms = int(t * 1000)
                chains.append(f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo,"
                              f"adelay={ms}|{ms},volume={vol:.2f}[a{i}]")
                labels.append(f"[a{i}]")
            graph = ";".join(chains) + f";{''.join(labels)}amix=inputs={len(sounds)}:normalize=0:duration=longest[aout]"
            args += ["-filter_complex", graph, "-map", "0:v", "-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", "-t", f"{duration:.3f}", out]
        res = run_ffmpeg(args)
        if res.returncode == 0:
            os.remove(tmp)
            self.on_done(f"gravação salva: recordings/{os.path.basename(out)}")
        else:
            self.on_done(f"erro ao finalizar gravação (vídeo bruto em {os.path.basename(tmp)})")
            print(res.stderr[-2000:])

    def wait(self):
        for th in self._threads:
            th.join()
