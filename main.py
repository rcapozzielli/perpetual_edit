"""perpetual_edit — observa a webcam e dispara edits quando reconhece gatilhos.

Estados: WAITING -> (gatilho) -> PRE_EDIT (som curto) -> PLAYING (clipe) -> WAITING
"""
import argparse
import json
import os
import random
import threading
import time
import traceback

import cv2
import numpy as np

from perpetual_edit.archive import ReplayArchive
from perpetual_edit.audio import AudioBus
from perpetual_edit.capture import Camera
from perpetual_edit.config import load_config
from perpetual_edit.detection import Detector
from perpetual_edit.hud import HUD
from perpetual_edit.player import EditPlayer, Popup, Slot, composite
from perpetual_edit.recorder import SessionRecorder
from perpetual_edit.template import ReplayBuffer, TemplatePlayer
from perpetual_edit.training import (CUSTOM_DIR, CustomModels, SampleCollector, create_model, latest_session,
                                     is_user_model, remove_model, sanitize, train_model, trash_samples)
from perpetual_edit.triggers import EVALUATORS, TriggerEngine

WINDOW = "MONITOR"
KEY_UP, KEY_DOWN = 2490368, 2621440  # setas no cv2.waitKeyEx (Windows)


class App:
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.cfg, warnings = load_config(config_path)
        self.hud = HUD()
        for w in warnings + self._type_warnings(self.cfg):
            self.hud.notify(w)
        self.bus = AudioBus()
        # O replay guarda sempre os últimos segundos da câmera (templates com "source": "replay").
        self.replay = ReplayBuffer(seconds=self._replay_seconds(self.cfg))
        self.clip_player = EditPlayer(self.bus)
        self.archive = ReplayArchive(self.cfg)  # replays de edits anteriores, para misturar
        if self.archive.clear_on_exit:
            self.archive.clear()  # sobra de uma sessão que não fechou direito
        self.template_player = TemplatePlayer(self.bus, self.replay, self.archive)
        self.player = self.clip_player  # o player do edit atual
        self.recorder = SessionRecorder(self.bus, on_done=self.hud.notify)
        self.engine = TriggerEngine(self.cfg)
        self.popup = Popup(self.cfg["popup"])
        self.slot = Slot(self.cfg["slot"])
        self.custom = CustomModels({}, self.cfg["smoothing"])
        self._load_custom()
        self.collector = SampleCollector(self.cfg["custom_models"])
        self._train_thread = None
        self._train_result = None
        # O import do scikit-learn leva alguns segundos: fazemos já, antes do loop
        # da câmera começar, para não travar a imagem no primeiro treino.
        threading.Thread(target=lambda: __import__("sklearn.linear_model"), daemon=True).start()
        self._warm_up()
        print("[app] carregando modelos do MediaPipe...")
        self.detector = Detector(self.cfg["detect_width"], self.cfg["smoothing"])
        self.camera = Camera(self.cfg["camera_index"], self.cfg["camera_width"],
                             self.cfg["camera_height"], self.cfg["mirror"], self.cfg["camera_backend"])
        self.mode = "WAITING"
        self.current_trigger = None
        self.current_edit = None
        self.pre_t0 = 0.0
        self.pre_duration = 0.0
        self.debug = False
        self.prompt = None  # campo de texto aberto (criar modelo), ou None
        self.slot_clip = None  # quadro do edit a desenhar na janela (modo slot)
        self.menu = None  # posição do cursor no menu de gatilhos (tecla E), ou None
        self.delete_armed = False  # apertou X no treino: a próxima tecla 1-9 escolhe a classe
        self.show_panel = True
        self.state_path = os.path.join(CUSTOM_DIR, "state.json")
        self._load_state()
        self.fps = 0.0

    @staticmethod
    def _type_warnings(cfg):
        return [f"gatilho '{n}': type '{t['type']}' desconhecido" for n, t in cfg["triggers"].items()
                if t["type"] not in EVALUATORS]

    def _load_custom(self):
        """(Re)carrega os modelos personalizados e avisa sobre os que faltam treinar."""
        missing = self.custom.load(self.cfg["custom_models"], self.cfg["smoothing"])
        used = {t.get("model") for t in self.cfg["triggers"].values() if t["type"] == "custom" and t["enabled"]}
        for m in missing:
            if m in used:
                self.hud.notify(f"modelo '{m}' ainda não treinado (T para treinar)")

    def _warm_up(self):
        """Extrai/carrega o áudio de todos os edits e os sons de pré/pós."""
        for name, e in self.cfg["edits"].items():
            try:
                self._player_for(e).prepare(e)
            except Exception as exc:
                self.hud.notify(f"edit {name}: {exc}")
        for key in ("pre_edit_sound", "post_edit_sound"):
            self.bus.preload(self.cfg[key])

    @staticmethod
    def _replay_seconds(cfg):
        """Quanto de câmera guardar: o maior (antes do gatilho + loading) entre os templates."""
        need = [e["spec"].get("replay", {}).get("before_s", 1.5) + e["pre_edit"]["duration"]
                for e in cfg["edits"].values() if "template" in e]
        return max(need, default=0.0) + 0.5

    def _player_for(self, edit_cfg):
        return self.template_player if "template" in edit_cfg else self.clip_player

    def reload(self):
        try:
            cfg, warnings = load_config(self.config_path)
        except Exception as exc:
            self.hud.notify(f"config inválido: {exc}")
            return
        self.cfg = cfg
        self.replay.seconds = self._replay_seconds(cfg)
        self.archive.configure(cfg)
        self.bus.clear_cache()
        self.engine.reconfigure(cfg)
        self.popup.cfg = cfg["popup"]
        self.slot.cfg = cfg["slot"]
        self.detector.set_smoothing(cfg["smoothing"])
        self._load_custom()
        self.collector.set_models(cfg["custom_models"])
        self._warm_up()
        warnings += self._type_warnings(cfg)
        for w in warnings:
            self.hud.notify(w)
        if not warnings:
            self.hud.notify("config recarregado")

    # ---------------- fluxo de um edit ----------------
    def start_edit(self, trigger_name, edit_name, label):
        edit_cfg = self.cfg["edits"][edit_name]
        self.player.stop()  # se outro edit ainda estiver aberto, fecha antes de trocar
        self.player = self._player_for(edit_cfg)
        try:
            self.player.load(edit_name, edit_cfg)
        except Exception as exc:
            self.hud.notify(str(exc))
            return
        self.current_trigger, self.current_edit = trigger_name, edit_name
        self.popup.rect = None  # posição nova a cada edit (sorteada no 1º frame)
        self.slot.start()  # sorteia se este edit usa a janela grande
        self.hud.show_label(label, self.cfg["label_duration_s"])
        pre = edit_cfg["pre_edit"]
        if pre["enabled"]:
            self.bus.play(pre["sound"], self.cfg["volume"]["sfx"])
            self.pre_duration = pre["duration"]
        else:
            self.pre_duration = 0.0
        self.pre_t0 = time.perf_counter()
        self.mode = "PRE_EDIT"
        print(f"[app] TRIGGER {trigger_name} -> {edit_name}")

    def manual_edit(self, edit=None):
        """Edit na hora (sem gatilho). Sem `edit`, sorteia um; com nome, toca aquele."""
        if self.mode != "WAITING" or not self.cfg["edits"]:
            return
        if edit is not None and edit not in self.cfg["edits"]:
            print(f"[app] edit '{edit}' não existe (rode tools\\import_template.py e aperte R)")
            return
        edit = edit or random.choice(list(self.cfg["edits"]))
        self.start_edit("manual", edit, "manual")

    def finish_edit(self):
        self.player.stop()
        self.bus.play(self.cfg["post_edit_sound"], self.cfg["volume"]["sfx"])
        self.engine.edit_finished()
        self.mode = "WAITING"
        self.current_trigger = self.current_edit = None

    def step(self, frame, features):
        """Avança a máquina de estados e devolve a imagem da câmera composta."""
        fired = self.engine.update(features, can_fire=self.mode == "WAITING")
        if fired:
            self.start_edit(fired, self.engine.choose_edit(fired), self.cfg["triggers"][fired]["label"])

        if self.mode == "PRE_EDIT" and time.perf_counter() - self.pre_t0 >= self.pre_duration:
            self.player.play(self.cfg["volume"]["edit"])
            self.mode = "PLAYING"

        out = frame
        self.slot_clip = None
        if self.mode == "PLAYING":
            clip = self.player.frame()
            if clip is None:
                self.finish_edit()
            elif self.cfg["overlay_mode"] == "slot":
                self.slot_clip = clip  # desenhado no loop, por cima do HUD (a janela cobre a caixa do rosto)
            elif self.cfg["overlay_mode"] == "popup":
                if self.popup.rect is None:
                    self.popup.start(frame.shape, clip.shape[1] / clip.shape[0], features.face_bbox)
                out = self.popup.draw(frame, clip, self.player.elapsed, self.player.duration - self.player.elapsed)
            else:
                out = composite(frame, clip, self.cfg["overlay_mode"], self.cfg["pip"])
        return out

    def state_info(self):
        if self.mode == "WAITING":
            return {"mode": "WAITING", "status": "WAITING...", "detail": ""}
        if self.mode == "PRE_EDIT":
            return {"mode": "PRE_EDIT", "status": f"TRIGGER: {self.current_trigger}", "detail": f"-> {self.current_edit}"}
        return {"mode": "PLAYING", "status": "PLAYING",
                "detail": f"{self.current_edit}  {self.player.elapsed:4.1f}/{self.player.duration:4.1f}s"}

    # ---------------- loop ----------------
    def run(self, max_seconds=None):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        frame_id, t_last = 0, time.perf_counter()
        t_begin = t_last
        n_frames = 0
        shown_w = None
        try:
            while True:
                frame_id, frame = self.camera.read(frame_id)
                if frame is None:
                    self.hud.notify("sem imagem da câmera...")
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        break
                    continue
                frame = frame.copy()
                features = self.detector.process(frame)
                self.custom.update(frame, features)
                self._check_training_done()
                if self.mode == "TRAIN":
                    self.collector.capture(frame, features)  # antes de desenhar o HUD por cima
                self.replay.push(frame, features)
                out = self.step(frame, features)
                fullscreen_clip = self.mode == "PLAYING" and self.cfg["overlay_mode"] == "fullscreen"
                rec_s = self.recorder.elapsed if self.recorder.active else None
                slot_mode = self.cfg["overlay_mode"] == "slot"
                # no modo slot o rótulo fica centralizado à esquerda da janela grande
                label_cx = self.slot.rect(out.shape, big=True)[0] // 2 if slot_mode else None
                self.hud.draw_overlay(out, features, show_face_box=not fullscreen_clip, recording_s=rec_s,
                                      label_cx=label_cx)
                progress = None
                if self.mode == "PRE_EDIT":
                    progress = (time.perf_counter() - self.pre_t0) / max(self.pre_duration, 1e-6)
                if self.slot_clip is not None:
                    out = self.slot.draw_clip(out, self.slot_clip)
                elif slot_mode and self.mode in ("WAITING", "PRE_EDIT"):
                    self.slot.draw_idle(out, progress)  # WAITING... / EDITING... (entra na gravação)
                elif progress is not None and self.pre_duration > 0:
                    # aviso visual de "loading" (entra na gravação, faz parte do edit)
                    self.hud.draw_loading(out, progress, self.player is self.template_player)
                self.recorder.write(out)  # grava antes do debug: o vídeo sai limpo
                if self.debug and not fullscreen_clip:
                    self.hud.draw_debug(out, features)
                panel = None
                if self.menu is not None:
                    panel = self.hud.draw_trigger_menu(out.shape[0], self.cfg, self.engine, self.menu)
                elif self.mode == "TRAIN":
                    self.hud.draw_train_overlay(out, self.collector)
                    panel = self.hud.draw_train_panel(out.shape[0], self.collector,
                                                      features.custom.get(self.collector.model), self.fps)
                    if self.prompt is not None:
                        self.hud.draw_prompt(out, self.prompt["label"], self.prompt["text"])
                elif self.show_panel:
                    panel = self.hud.draw_panel(out.shape[0], self.state_info(), self.engine, self.cfg,
                                                self.fps, self.debug)
                if panel is None:
                    self.hud.draw_toast(out)
                    shown = out
                else:
                    shown = np.hstack([out, panel])
                if shown.shape[1] != shown_w:  # painel apareceu/sumiu: ajusta a janela sem esticar
                    shown_w = shown.shape[1]
                    cv2.resizeWindow(WINDOW, shown.shape[1], shown.shape[0])
                cv2.imshow(WINDOW, shown)

                now = time.perf_counter()
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-6, now - t_last)) if self.fps else 1.0 / max(1e-6, now - t_last)
                t_last = now
                n_frames += 1

                raw = cv2.waitKeyEx(1)  # Ex: entrega também as setas
                key = raw & 0xFF if raw != -1 else 255
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if self.prompt is not None:  # digitando: todas as teclas vão para o campo de texto
                    if key != 255:
                        self.prompt_key(key)
                elif self.menu is not None:  # menu aberto: as teclas são do menu
                    if raw != -1:
                        self.menu_key(raw, key)
                elif key in (ord("q"), ord("Q"), 27):
                    break
                elif key in (ord("d"), ord("D")):
                    self.debug = not self.debug
                elif key in (ord("r"), ord("R")):
                    self.reload()
                elif key in (ord("g"), ord("G")):
                    self.toggle_recording(out.shape[1], out.shape[0])
                elif key in (ord("t"), ord("T")):
                    self.toggle_train_mode()
                elif key in (ord("e"), ord("E")) and self.mode != "TRAIN":
                    self.menu = 0
                elif key in (ord("h"), ord("H")):
                    self.toggle_panel()
                elif self.mode == "TRAIN":
                    self.train_key(key)
                elif key == ord(" "):
                    self.manual_edit()
                elif key in (ord("l"), ord("L")):  # TEMPORÁRIO: testar o template de tela verde
                    self.manual_edit("lossing")

                if max_seconds and now - t_begin > max_seconds:
                    break
        finally:
            elapsed = time.perf_counter() - t_begin
            print(f"[app] {n_frames} frames em {elapsed:.1f}s -> média {n_frames / max(elapsed, 1e-6):.1f} fps")
            self.close()

    # ---------------- menu de gatilhos e painel ----------------
    def menu_key(self, raw, key):
        names = list(self.cfg["triggers"])
        if key in (ord("e"), ord("E"), 27):
            self.menu = None
        elif not names:
            return
        elif raw == KEY_UP or key in (ord("w"), ord("W")):
            self.menu = (self.menu - 1) % len(names)
        elif raw == KEY_DOWN or key in (ord("s"), ord("S")):
            self.menu = (self.menu + 1) % len(names)
        elif key in (32, 13):  # ESPAÇO / ENTER
            name = names[self.menu]
            self.set_muted([name], self.engine.is_on(name))
        elif key in (ord("a"), ord("A")):
            # algum ligado -> desliga todos; senão liga todos
            self.set_muted(names, any(self.engine.is_on(n) for n in names))

    def set_muted(self, names, muted):
        for n in names:
            if not self.cfg["triggers"][n]["enabled"]:
                if len(names) == 1:
                    self.hud.notify(f"'{n}' está desligado no config.json")
                continue
            if muted:
                self.engine.muted.add(n)
            else:
                self.engine.muted.discard(n)
                self.engine.states[n].armed = False  # não dispara na hora só porque religou
        self._save_state()

    def toggle_panel(self):
        self.show_panel = not self.show_panel
        self.hud.notify("painel visível" if self.show_panel else "painel escondido (H mostra)")
        self._save_state()

    def _load_state(self):
        """custom/state.json: o que você escolheu na tela (gatilhos desligados, painel)."""
        try:
            with open(self.state_path, "r", encoding="utf-8-sig") as f:
                st = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self.engine.muted = set(st.get("muted", []))
        self.show_panel = bool(st.get("show_panel", True))

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"muted": sorted(self.engine.muted), "show_panel": self.show_panel}, f, indent=2)
            os.replace(tmp, self.state_path)
        except OSError as exc:
            self.hud.notify(f"não consegui salvar custom/state.json ({exc})")

    # ---------------- modo treino ----------------
    def toggle_train_mode(self):
        if self.mode == "TRAIN":
            self.collector.stop()
            self.engine.disarm_all()  # não disparar com a pose que estava treinando
            self.mode = "WAITING"
            self.hud.notify("modo treino desligado")
        elif self.mode == "WAITING":
            self.collector.refresh()
            self.mode = "TRAIN"
            self.hud.notify("modo treino: 1-9 grava uma classe")
        else:
            self.hud.notify("espere o edit terminar para entrar no treino")

    def train_key(self, key):
        if key == 255:
            return
        if self.delete_armed:
            self.delete_armed = False
            if key == ord("0"):
                self.confirm_delete_model()
            elif ord("1") <= key <= ord("9") and key - ord("1") < len(self.collector.classes):
                self.confirm_delete(self.collector.classes[key - ord("1")])
            else:
                self.hud.notify("apagar cancelado")
            return
        if ord("1") <= key <= ord("9"):
            self.collector.toggle(key - ord("1"))
        elif key in (ord("x"), ord("X")) and self.collector.model:
            self.collector.stop()
            self.delete_armed = True
            self.hud.notify(f"apagar: número da classe (1-{len(self.collector.classes)}) "
                            "ou 0 para o modelo inteiro", 8)
        elif key in (ord("u"), ord("U")) and self.collector.model:
            self.collector.stop()
            last = latest_session(self.collector.model, self.collector.classes)
            if last is None:
                self.hud.notify("nenhuma amostra gravada neste modelo")
            else:
                self.confirm_delete(*last)
        elif key == 9:  # TAB
            self.collector.next_model()
        elif key == 13:  # ENTER
            self.start_training()
        elif key in (ord("n"), ord("N")):
            self.new_model_dialog()

    def confirm_delete(self, cls, session=None):
        """Pergunta antes de mover amostras para custom/_lixeira (classe inteira ou uma sessão)."""
        model = self.collector.model
        if session:
            n = self.collector.session_size(cls, session)
            what = f"a última sessão de '{cls}' ({n} amostras)"
        else:
            n = self.collector.counts().get(cls, 0)
            what = f"TODAS as {n} amostras de '{cls}'"
        if n == 0:
            self.hud.notify(f"'{cls}' não tem amostras")
            return

        def done(text):
            if text.strip().lower() not in ("s", "sim"):
                self.hud.notify("apagar cancelado")
                return
            moved = trash_samples(model, cls, session)
            self.collector.refresh()
            self.hud.notify(f"{moved} amostras de '{cls}' foram para custom/_lixeira "
                            "(ENTER treina de novo sem elas)", 8)

        self.ask(f"apagar {what}? (s = sim)", "", done)

    def confirm_delete_model(self):
        """X 0: apaga o modelo selecionado inteiro (pede para digitar o nome)."""
        if self._train_thread:
            self.hud.notify("espere o treino atual terminar")
            return
        name = self.collector.model
        if not is_user_model(name):
            self.hud.notify(f"'{name}' está declarado no config.json: para apagar, tire de lá", 8)
            return

        def done(text):
            if sanitize(text) != name:
                self.hud.notify("nome diferente: nada apagado")
                return
            try:
                dst = remove_model(name)
            except Exception as exc:
                self.hud.notify(f"não apaguei: {exc}", 8)
                return
            self.reload()
            self.hud.notify(f"modelo '{name}' e o gatilho dele foram para custom/_lixeira/{name}", 8)

        self.ask(f"apagar o modelo '{name}' inteiro? digite o nome para confirmar", "", done)

    # ---------------- campo de texto (criar modelo) ----------------
    def ask(self, label, default, on_done):
        self.prompt = {"label": label, "text": default, "on_done": on_done}

    def prompt_key(self, key):
        p = self.prompt
        if key == 13:  # ENTER confirma
            self.prompt = None  # limpa antes: o callback pode abrir outro campo
            p["on_done"](p["text"])
        elif key == 27:  # ESC cancela
            self.prompt = None
            self.hud.notify("cancelado")
        elif key == 8:  # BACKSPACE
            p["text"] = p["text"][:-1]
        elif 32 <= key <= 126 and len(p["text"]) < 60:  # o OpenCV só entrega ASCII (sem acentos)
            p["text"] += chr(key)

    def new_model_dialog(self):
        """N no modo treino: pede o nome e as classes e cria o modelo em custom/models.json."""
        if self._train_thread:
            self.hud.notify("espere o treino atual terminar")
            return
        self.collector.stop()

        def got_name(text):
            name = sanitize(text)
            if not name:
                self.hud.notify("nome vazio, nada criado")
                return
            if name in self.cfg["custom_models"]:
                self.hud.notify(f"já existe um modelo '{name}'")
                return

            def got_classes(text2):
                classes = [sanitize(c) for c in text2.split(",") if sanitize(c)]
                if not 2 <= len(classes) <= 9 or len(set(classes)) != len(classes):
                    self.hud.notify("precisa de 2 a 9 classes diferentes, separadas por vírgula")
                    return
                try:
                    create_model(name, classes)
                except Exception as exc:
                    self.hud.notify(f"não consegui criar: {exc}")
                    return
                self.reload()
                self.collector.select(name)
                self.hud.notify(f"'{name}' criado: grave com 1-{len(classes)} e treine com ENTER", 8)

            self.ask(f"classes de '{name}' (separadas por vírgula)", f"normal, {name}", got_classes)

        self.ask("nome do novo modelo", "", got_name)

    def start_training(self):
        if self._train_thread or not self.collector.model:
            return
        self.collector.stop()
        name = self.collector.model
        cfg = self.cfg["custom_models"][name]
        self.hud.notify(f"treinando '{name}'...", 60)

        def job():
            try:
                self._train_result = train_model(name, cfg)
            except Exception as exc:
                self._train_result = f"erro no treino: {exc}"

        self._train_thread = threading.Thread(target=job, daemon=True)
        self._train_thread.start()

    def _check_training_done(self):
        """Quando o treino (em outra thread) termina, recarrega os modelos aqui na thread principal."""
        if self._train_thread and not self._train_thread.is_alive():
            self._train_thread = None
            self.custom.load(self.cfg["custom_models"], self.cfg["smoothing"])
            self.hud.notify(self._train_result, 8)

    def toggle_recording(self, width, height):
        if self.recorder.active:
            self.recorder.stop()
            self.hud.notify("finalizando gravação...")
        else:
            self.recorder.start(width, height)
            self.hud.notify("gravando sessão (G para parar)")

    def close(self):
        if self.recorder.active:
            self.recorder.stop()
        if self.recorder._threads:
            print("[app] aguardando a gravação terminar de ser salva...")
        self.recorder.wait()
        self.clip_player.stop()
        self.template_player.stop()
        if self.archive.clear_on_exit:
            n = self.archive.clear()
            print(f"[app] {n} replay(s) da sessão apagado(s)")
        self.camera.release()
        self.detector.close()
        self.bus.close()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="perpetual_edit")
    ap.add_argument("--config", default=None, help="caminho do config.json")
    ap.add_argument("--seconds", type=float, default=None, help="fecha sozinho após N segundos (teste)")
    args = ap.parse_args()
    try:
        App(args.config).run(args.seconds)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
