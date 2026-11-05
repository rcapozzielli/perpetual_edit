"""Testes sem câmera: config, gatilhos simulados, player e detector.

Uso:  python tests/test_core.py
"""
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perpetual_edit.config import load_config, parse_time  # noqa: E402
from perpetual_edit.detection import Features  # noqa: E402
from perpetual_edit.triggers import TriggerEngine  # noqa: E402


def test_parse_time():
    assert parse_time("00:04.50") == 4.5
    assert parse_time("01:02:03.5") == 3723.5
    assert parse_time(2) == 2.0
    assert parse_time(None) is None


def unexpected(warnings):
    """Tira os avisos normais de um clone limpo (os mp4 do CapCut não vão para o repositório)."""
    return [w for w in warnings
            if not ("template inválido" in w and "não encontrado" in w)
            and not w.startswith("nenhum template disponível")]


def has_templates(cfg):
    return any("template" in e for e in cfg["edits"].values())


def test_config():
    cfg, warnings = load_config()
    assert not unexpected(warnings), warnings
    assert cfg["edits"]
    kinds = {"template" in e for e in cfg["edits"].values()}
    assert len(kinds) == 1, kinds  # só templates ou, no clone limpo, só os clipes de teste
    assert cfg["global_cooldown_s"] == 15
    assert cfg["triggers"]["thumbs_up"]["type"] == "gesture"
    return cfg


def test_triggers(cfg):
    eng = TriggerEngine(cfg)
    mouth = Features(face_found=True, blendshapes={"jawOpen": 0.9})
    hold = cfg["triggers"]["mouth_open"]["hold_frames"]
    t = 100.0
    fired = [eng.update(mouth, True, t + i * 0.03) for i in range(hold)]
    assert fired[:-1] == [None] * (hold - 1), "disparou antes do hold"
    assert fired[-1] == "mouth_open", fired
    # mesma pose continua: não pode disparar de novo (rearme por borda)
    eng.edit_finished(t + 10)
    assert eng.update(mouth, True, t + 30) is None
    # fecha a boca, reabre: dispara de novo
    eng.update(Features(face_found=True, blendshapes={"jawOpen": 0.0}), True, t + 31)
    out = [eng.update(mouth, True, t + 32 + i * 0.03) for i in range(hold)]
    assert out[-1] == "mouth_open", out
    # cooldown global: logo após o fim de um edit não dispara
    eng.edit_finished(t + 40)
    eng.update(Features(face_found=True), True, t + 40.1)
    out = [eng.update(mouth, True, t + 40.2 + i * 0.03) for i in range(hold)]
    assert out[-1] is None
    # gesto
    thumb = Features(face_found=True, gesture="Thumb_Up", gesture_score=0.9)
    eng2 = TriggerEngine(cfg)
    out = [eng2.update(thumb, True, 200 + i * 0.03) for i in range(cfg["triggers"]["thumbs_up"]["hold_frames"])]
    assert out[-1] == "thumbs_up", out
    # sem repetir o último edit
    picks = [eng.choose_edit("mouth_open") for _ in range(20)]
    assert all(a != b for a, b in zip(picks, picks[1:])), picks
    # desligado pelo menu (tecla E): não dispara; religado volta a disparar
    eng3 = TriggerEngine(cfg)
    eng3.muted.add("thumbs_up")
    out = [eng3.update(thumb, True, 300 + i * 0.03) for i in range(30)]
    assert set(out) == {None} and not eng3.is_on("thumbs_up"), out
    eng3.muted.discard("thumbs_up")
    out = [eng3.update(thumb, True, 301 + i * 0.03) for i in range(cfg["triggers"]["thumbs_up"]["hold_frames"])]
    assert out[-1] == "thumbs_up", out


def test_popup():
    import random
    from perpetual_edit.player import Popup
    cam = np.full((720, 1280, 3), 40, np.uint8)
    clip = np.full((1920, 1080, 3), (0, 0, 255), np.uint8)  # edit 9:16 vermelho
    face = (0.4, 0.3, 0.6, 0.7)  # rosto no meio
    pop = Popup({"height": 0.55, "tilt_deg": 4, "border": 6, "margin": 30})
    spots = set()
    for seed in range(30):
        pop.start(cam.shape, 1080 / 1920, face, random.Random(seed))
        x, y, w, h = pop.rect
        assert 30 <= x and x + w <= 1250 and 30 <= y and y + h <= 690, pop.rect
        assert x + w <= 0.4 * 1280 or x >= 0.6 * 1280, f"cobriu o rosto: {pop.rect}"
        assert abs(h - 0.55 * 720) < 2 and abs(w / h - 1080 / 1920) < 0.02
        spots.add((x // 100, y // 100))
    assert len(spots) >= 5, f"posições pouco variadas: {spots}"
    # meio do edit: card inteiro na tela; começo e fim: menor (animação de entrada/saída)
    mid = pop.draw(cam, clip, 1.0, 2.0)
    red = lambda img: int(((img[..., 2] > 200) & (img[..., 0] < 60)).sum())
    full = red(mid)
    assert abs(full - w * h) < 0.08 * w * h, (full, w * h)
    assert red(pop.draw(cam, clip, 0.03, 3.0)) < 0.3 * full
    assert red(pop.draw(cam, clip, 3.0, 0.05)) < 0.3 * full
    assert red(pop.draw(cam, clip, 3.0, 0.0)) == 0
    assert (cam == 40).all(), "draw não pode alterar a imagem da câmera"


def test_phase2_evaluators():
    from perpetual_edit.triggers import EVALUATORS as E

    face = dict(face_found=True, face_bbox=(0.4, 0.3, 0.6, 0.7), aspect=16 / 9)
    assert E["eyes_closed"](Features(**face, blendshapes={"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.8}), {}) == 0.8
    assert E["eyes_closed"](Features(**face, blendshapes={"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}), {}) == 0.1
    turned = Features(**face, yaw_deg=-30)
    assert E["head_turn"](turned, {"side": "left"}) == 30
    assert E["head_turn"](turned, {"side": "right"}) == 0
    assert E["head_turn"](turned, {}) == 30
    hand_on = np.array([[0.5, 0.5, 0]] * 21, np.float32)
    hand_far = np.array([[0.95, 0.5, 0]] * 21, np.float32)
    assert E["hand_near_face"](Features(**face, hands=[hand_on]), {}) == 1.0
    assert E["hand_near_face"](Features(**face, hands=[hand_far]), {}) == 0.0
    assert E["hand_near_face"](Features(**face), {}) == 0.0
    assert E["face_lost"](Features(face_missing_s=4.0), {}) == 4.0
    assert E["face_back"](Features(**face, face_back_after_s=2.5), {}) == 2.5


def test_custom_trigger():
    from perpetual_edit.training import BS_NAMES, align_face
    from perpetual_edit.triggers import EVALUATORS as E

    assert len(BS_NAMES) == 52
    lms = np.zeros((478, 3), np.float32) + 0.5
    lms[[33, 133], 0], lms[[362, 263], 0] = 0.45, 0.55
    assert align_face(np.zeros((720, 1280, 3), np.uint8), lms).shape == (128, 128, 3)

    cfg = {"model": "glasses", "from": "com_oculos", "to": "sem_oculos"}

    def state(stable, previous, since):
        return Features(custom={"glasses": {"probs": {"com_oculos": 0.1, "sem_oculos": 0.9},
                                            "stable": stable, "previous": previous,
                                            "since": since, "fresh_s": 2.0}})
    assert E["custom"](state("sem_oculos", "com_oculos", 0.5), cfg) == 0.9   # acabou de tirar
    assert E["custom"](state("sem_oculos", "com_oculos", 5.0), cfg) == 0.0   # já faz tempo
    assert E["custom"](state("sem_oculos", None, 0.5), cfg) == 0.0           # começou sem óculos
    assert E["custom"](Features(), cfg) == 0.0                               # modelo não treinado


def test_template(cfg):
    from perpetual_edit.audio import AudioBus
    from perpetual_edit.template import EFFECTS, ReplayBuffer, TemplatePlayer

    bus = AudioBus()
    names = [n for n, e in cfg["edits"].items() if "template" in e]
    assert names
    for name in names:
        edit = cfg["edits"][name]
        spec = edit["spec"]
        assert all(fx["type"] in EFFECTS for s in spec["segments"] for fx in s.get("effects", []))
        assert edit["pre_edit"]["duration"] == 2.0 and edit["pre_edit"]["sound"]
        replay = ReplayBuffer(4.0)
        cam = np.full((720, 1280, 3), 120, np.uint8)
        # Simula: gatilho há 2 s (loading) e câmera desde 1,5 s antes dele.
        trigger = time.perf_counter() - 2.0
        for i in range(105):
            replay.push(cam, Features(), trigger - 1.5 + i / 30)
        pl = TemplatePlayer(bus, replay)
        pl.load(name, edit)
        pl.t_trigger = trigger
        pl.play(0.0)
        assert pl.wav, "áudio do template não extraído"
        assert len(pl._clip) >= 100 and pl._times[0] < -1.4 and pl._times[-1] > 1.9, (len(pl._clip), pl._times[[0, -1]])
        for seg in spec["segments"]:
            ts = (seg["start"] + seg["end"]) / 2
            pl.t0 = time.perf_counter() - ts
            img = pl.frame()
            w, h = spec["size"]
            assert img is not None and img.shape == (h, w, 3), (name, ts, None if img is None else img.shape)
        pl.t0 = time.perf_counter() - spec["duration"] - 0.1
        assert pl.frame() is None and not pl.playing
    bus.close()
    print(f"  templates: {', '.join(names)}")


def test_auto_templates(cfg):
    templates = sorted(n for n, e in cfg["edits"].items() if "template" in e)
    for name, t in cfg["triggers"].items():  # todo gatilho sorteia entre todos os templates
        assert sorted(t["edits"]) == templates, (name, t["edits"])


def test_demo_fallback(cfg):
    clips = sorted(cfg["edits"])
    assert clips == ["edit_1", "edit_2", "edit_3"], clips
    for name, t in cfg["triggers"].items():  # sem template, "@templates" usa os clipes de teste
        assert sorted(t["edits"]) == clips, (name, t["edits"])


def test_archive(cfg):
    import shutil
    import tempfile

    from perpetual_edit.archive import ReplayArchive
    from perpetual_edit.audio import AudioBus
    from perpetual_edit.template import ReplayBuffer, TemplatePlayer

    tmp = tempfile.mkdtemp()
    try:
        arch = ReplayArchive({"archive": {"dir": tmp, "max_clips": 2, "mix": 1.0}})
        lms = np.full((478, 2), 0.5, np.float32)
        for k in range(3):  # 3 replays com max_clips=2: o mais antigo sai
            clip = [(-1.5 + i / 30, np.full((600, 600, 3), 40 * k, np.uint8), lms) for i in range(105)]
            arch._save(clip)
        assert len(arch.list()) == 2
        part = arch.load_range(arch.list()[0], 0.0, 0.5)
        assert 14 <= len(part) <= 19 and part[0][2] is not None
        arch.save_async([(0.0, np.zeros((600, 600, 3), np.uint8), None)] * 10)  # sem rosto: ignorado
        time.sleep(0.3)
        assert len(arch.list()) == 2

        # com mix=1.0, todo corte rápido (depois do 1º trecho) usa replay antigo
        name = next(n for n, e in cfg["edits"].items() if len(e["spec"]["segments"]) > 5)
        replay = ReplayBuffer(4.0)
        trigger = time.perf_counter() - 2.0
        for i in range(105):
            replay.push(np.full((720, 1280, 3), 200, np.uint8), Features(), trigger - 1.5 + i / 30)
        bus = AudioBus()
        pl = TemplatePlayer(bus, replay, arch)
        t_load = time.perf_counter()
        pl.load(name, cfg["edits"][name])
        pl._archive_thread.join()  # no app isso acontece durante os 2 s de loading
        load_s = time.perf_counter() - t_load
        assert load_s < 2.0, f"carregar replays antigos levou {load_s:.2f}s (mais que o loading)"
        pl.t_trigger = trigger
        pl.play(0.0)
        segs = pl.spec["segments"]
        expected = {i for i, s in enumerate(segs)
                    if i > 0 and s["source"] in ("replay", "keyed") and s["end"] - s["start"] < 1.5}
        assert set(pl._seg_clips) == expected, (sorted(pl._seg_clips), sorted(expected))
        pl.stop()
        bus.close()
        print(f"  arquivo: {len(expected)} cortes de '{name}' com replays antigos, carregados em {load_s:.2f}s")

        # ao fechar o app: apaga os replays (inclusive um que ainda está sendo salvo),
        # mas não mexe em arquivos que não são nossos
        assert arch.clear_on_exit
        open(os.path.join(tmp, "meu_arquivo.txt"), "w").close()
        arch.save_async([(0.0, np.zeros((600, 600, 3), np.uint8), lms)] * 60)
        assert arch.clear() == 2 and arch.list() == []  # max_clips=2: o novo empurrou um antigo
        assert os.listdir(tmp) == ["meu_arquivo.txt"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_importer_detect():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import import_template as it

    # 3 s a 30 fps: corte seco no frame 30, flash (que se apaga) no 60, preto final a partir do 80
    bright = np.r_[np.full(30, 90.0), np.full(30, 40.0), np.linspace(200, 60, 20), np.zeros(10)]
    diff = np.r_[0, np.abs(np.diff(bright))]
    cuts, black, _ = it.detect(30.0, bright, diff)
    assert [c for c, _ in cuts if c < 80] == [30, 60], cuts
    assert dict(cuts)[60] is True and dict(cuts)[30] is False
    assert black == 80
    spec = it.build("t", "x.mp4", 30.0, (1080, 1440), bright, diff)
    assert [s["start"] for s in spec["segments"]] == [0.0, 1.0, 2.0, 2.67]
    assert spec["segments"][-1]["source"] == "color"


def test_keyed():
    """Tela verde: só o verde vira a sua imagem; texto e emoji do template ficam."""
    import cv2

    from perpetual_edit.template import compose_keyed

    def slot_frame(value=230, hue=60):
        hsv = np.zeros((300, 300, 3), np.uint8)
        hsv[:, :] = (hue, 255, value)
        f = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        f[:60, :] = (40, 20, 20)                                        # faixa escura com texto
        cv2.putText(f, "LOSING", (10, 45), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
        cv2.circle(f, (150, 180), 30, (0, 0, 255), -1)                  # "emoji" vermelho no meio
        return f

    user = lambda size: np.full((size[1], size[0], 3), (200, 100, 0), np.uint8)  # azul
    out = compose_keyed(slot_frame(), user)
    assert (out[250, 250] == out[250, 250]).all() and abs(int(out[250, 250, 0]) - 200) < 20, out[250, 250]
    assert out[250, 250, 1] < 120                                       # verde trocado
    assert tuple(out[180, 150]) == (0, 0, 255)                          # emoji intacto
    assert tuple(out[20, 280]) == (40, 20, 20)                          # faixa do texto intacta
    assert out[40:60, 10:200].max() > 200                               # texto branco continua
    # a caixa da sua imagem é a do verde (abaixo da faixa)
    boxes = []
    compose_keyed(slot_frame(), lambda size: boxes.append(size) or user(size))
    assert boxes == [(300, 240)], boxes
    # template escureceu o verde pela metade -> sua imagem também escurece (com força "light")
    dark = compose_keyed(slot_frame(value=115), user)
    assert 0.55 < dark[250, 250, 0] / out[250, 250, 0] < 0.8, (dark[250, 250], out[250, 250])
    # sem verde: quadro original
    plain = np.full((300, 300, 3), 30, np.uint8)
    assert compose_keyed(plain, user) is plain


def test_importer_green():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import import_template as it

    fps = 30.0
    # Verdes A (tom ~47 depois do filtro) e B (~70) alternados.
    # 0-0.5 s sem verde; intro A 0.5-2.5 s com o tom "escorregando" no fade; 1 quadro sem verde;
    # B; corte direto (sem pausa) para A; pausa; B; e um flash (2 quadros sem verde) no meio dele
    hue = np.r_[np.full(15, -1.0), [52, 50, 48], np.full(57, 47.0), [-1],
                np.full(14, 70.0), np.full(14, 47.0), np.full(3, -1.0),
                np.full(7, 70.0), [-1, -1], np.full(7, 70.0), np.full(10, -1.0)]
    cover = np.where(hue < 0, 0.0, 1.0)
    runs = it.green_runs(fps, cover, hue)
    assert [(a, b) for a, b, _ in runs] == [(15, 75), (76, 90), (90, 104), (107, 123)], runs
    spec = it.build_green("t", "x.mp4", fps, (1080, 1080), cover, hue)
    segs = spec["segments"]
    assert [s["source"] for s in segs] == ["keyed"] * 4 and segs[0]["start"] == 0.0
    assert segs[-1]["end"] == spec["duration"]
    assert all(a["end"] == b["start"] for a, b in zip(segs, segs[1:]))   # sem buracos na linha do tempo
    assert segs[0]["replay_at"] < -1.5                                   # intro: replay inteiro
    # cada espaço mostra um momento diferente do gesto (mesmo os do mesmo tom)
    assert len({s["replay_at"] for s in segs[1:]}) == 3, [s["replay_at"] for s in segs]
    assert spec["imported"]["spaces"] == 4
    assert it.slug("capcut_video/green/lossing_green.mp4", green=True) == "lossing"


def test_new_models():
    import shutil
    import tempfile

    import cv2

    from perpetual_edit import training as tr

    assert tr.sanitize("Bebendo água!") == "bebendo_agua"
    assert tr.sanitize("  Com Fone ") == "com_fone"

    # rosto: bbox x 0.4-0.6; boca (13) no centro; laterais 234/454
    lms = np.full((478, 3), 0.5, np.float32)
    lms[:, 0] = np.linspace(0.4, 0.6, 478)
    lms[13, :2], lms[4, :2] = (0.5, 0.6), (0.5, 0.5)
    lms[234, :2], lms[454, :2] = (0.4, 0.5), (0.6, 0.5)
    lms[[33, 133], 0], lms[[362, 263], 0] = 0.45, 0.55
    face = dict(face_found=True, face_landmarks=lms, aspect=16 / 9)
    assert tr.hand_vector(Features(**face)).shape == (tr.HAND_DIM,)
    assert (tr.hand_vector(Features(**face)) == tr.empty_hand_vector()).all()
    on_mouth = np.tile(np.array([[0.5, 0.6, 0.0]], np.float32), (21, 1))
    v = tr.hand_vector(Features(**face, hands=[on_mouth]))
    assert v[0] == 1.0 and v[tr._PER_HAND - 3] < 0.05  # mão presente, colada na boca
    big = tr.align_face(np.zeros((720, 1280, 3), np.uint8), lms, scale=1.5)
    assert big.shape == (128, 128, 3)

    tmp = tempfile.mkdtemp()
    old_dir = tr.CUSTOM_DIR
    try:
        # cria o modelo e confere que o config enxerga modelo + gatilho
        path = os.path.join(tmp, "models.json")
        tr.create_model("bebendo_agua", ["normal", "bebendo_agua"], path)
        try:
            tr.create_model("bebendo_agua", ["a", "b"], path)
            raise AssertionError("deveria recusar nome repetido")
        except ValueError:
            pass
        cfg, warnings = load_config(user_models=path)
        assert not unexpected(warnings), warnings
        assert cfg["custom_models"]["bebendo_agua"]["crop_scale"] == 1.5
        trig = cfg["triggers"]["bebendo_agua"]
        assert trig["type"] == "custom" and trig["to"] == "bebendo_agua" and trig["edits"]
        assert "glasses" in cfg["custom_models"]  # os do config.json continuam

        # treino: mesma imagem nas duas classes, só a mão muda -> as mãos têm que separar
        tr.CUSTOM_DIR = tmp
        rng = np.random.default_rng(0)
        for cls, hands in (("normal", None), ("bebendo_agua", [on_mouth])):
            for s in range(2):
                d = os.path.join(tmp, "bebendo_agua", cls)
                os.makedirs(d, exist_ok=True)
                for i in range(15):
                    base = os.path.join(d, f"2026010100000{s}_{i:05d}")
                    cv2.imwrite(base + ".png", rng.integers(90, 110, (128, 128, 3), dtype=np.uint8))
                    np.save(base + ".npy", np.zeros(52, np.float32))
                    np.save(base + "_h.npy", tr.hand_vector(Features(**face, hands=hands or [])))
        msg = tr.train_model("bebendo_agua", cfg["custom_models"]["bebendo_agua"])
        assert "validação 100%" in msg, msg
        cm = tr.CustomModels({"bebendo_agua": cfg["custom_models"]["bebendo_agua"]}, smoothing=1.0)
        f = Features(**face, hands=[on_mouth])
        cm.update(np.full((720, 1280, 3), 100, np.uint8), f)
        assert f.custom["bebendo_agua"]["probs"]["bebendo_agua"] > 0.9, f.custom
        print(f"  modelo novo: {msg}")

        # apagar pelo app: U desfaz a última sessão, X apaga a classe (vão para a lixeira)
        last = tr.latest_session("bebendo_agua", ["normal", "bebendo_agua"])
        assert last[1] == "20260101000001", last  # a sessão mais recente
        assert tr.trash_samples("bebendo_agua", "bebendo_agua", last[1]) == 15
        assert tr.count_samples("bebendo_agua", ["bebendo_agua"]) == {"bebendo_agua": 15}
        assert tr.trash_samples("bebendo_agua", "normal") == 30
        assert tr.count_samples("bebendo_agua", ["normal"]) == {"normal": 0}
        trash = glob.glob(os.path.join(tmp, "_lixeira", "bebendo_agua", "*", "*", "*.png"))
        assert len(trash) == 45 and len(glob.glob(os.path.join(tmp, "_lixeira", "**", "*_h.npy"),
                                                  recursive=True)) == 45
        assert tr.trash_samples("bebendo_agua", "normal") == 0  # nada mais para mover

        # apagar o modelo inteiro (X 0): sai do models.json com o gatilho, pasta vai para a lixeira
        assert tr.is_user_model("bebendo_agua", path) and not tr.is_user_model("glasses", path)
        dst = tr.remove_model("bebendo_agua", path)
        assert not os.path.exists(os.path.join(tmp, "bebendo_agua"))
        assert os.path.exists(os.path.join(dst, "model.joblib"))
        assert os.path.exists(os.path.join(dst, "models_entry.json"))
        cfg2, _ = load_config(user_models=path)
        assert "bebendo_agua" not in cfg2["custom_models"] and "bebendo_agua" not in cfg2["triggers"]
        try:
            tr.remove_model("glasses", path)  # do config.json: não mexe
            raise AssertionError("não deveria apagar modelo do config.json")
        except ValueError:
            pass
    finally:
        tr.CUSTOM_DIR = old_dir
        shutil.rmtree(tmp, ignore_errors=True)


def test_slot():
    import random
    from perpetual_edit.player import Slot
    cam = np.full((720, 1280, 3), 120, np.uint8)
    slot = Slot({"corner": "top_right", "width": 0.28, "big_width": 0.55, "big_chance": 0.33,
                 "aspect": 1.4, "margin": 40, "top": 72})
    x, y, w, h = slot.rect(cam.shape)
    assert x + w == 1240 and y == 72 and abs(w / h - 1.4) < 0.02
    bx, by, bw, bh = slot.rect(cam.shape, big=True)
    assert bw > 1.8 * w and bx + bw == 1240 and by + bh <= 720
    # parado: janela escura com a mira; resto da imagem intacto
    idle = slot.draw_idle(cam.copy())
    assert idle[y + 5:y + h - 5, x + 5:x + 30].mean() < 60 and (idle[:, :x - 2] == 120).all()
    # edit 1:1 numa janela 1.4 -> preenche; 9:16 -> inteiro com fundo desfocado
    rng = random.Random(1)
    sizes = []
    for _ in range(30):
        slot.start(rng)
        sizes.append(slot.big)
    assert 3 <= sum(sizes) <= 20, sizes  # às vezes grande, na maioria pequena
    slot.big = False
    red = np.full((600, 600, 3), (0, 0, 255), np.uint8)
    out = slot.draw_clip(cam, red)
    inner = out[y + 4:y + h - 4, x + 4:x + w - 4]
    assert (inner[..., 2] > 200).mean() > 0.99 and (cam == 120).all()
    tall = np.full((1920, 1080, 3), (0, 0, 255), np.uint8)
    out = slot.draw_clip(cam, tall)
    inner = out[y + 4:y + h - 4, x + 4:x + w - 4]
    assert 0.3 < ((inner[..., 2] > 200) & (inner[..., 0] < 60)).mean() < 0.6


def test_face_cube():
    from perpetual_edit.hud import face_cube
    bbox, size = (0.4, 0.3, 0.6, 0.7), (1280, 720)
    p = face_cube(bbox, size, 0, 0, 0)
    front, back = p[:4], p[4:]
    assert abs(front[:, 0].mean() - 640) <= 1 and abs(front[:, 1].mean() - 360) <= 1  # frente sobre o rosto
    fw, bw = np.ptp(front[:, 0]), np.ptp(back[:, 0])
    assert abs(fw - 0.2 * 1280 * 1.1) < 25 and bw < fw  # de frente: o fundo aparece menor, centralizado
    assert abs(back[:, 0].mean() - 640) <= 1
    right = face_cube(bbox, size, 30, 0, 0)  # virou para a direita: o fundo vai para a esquerda
    assert right[4:, 0].mean() < 640 - 40
    down = face_cube(bbox, size, 0, 20, 0)  # olhou para baixo: o fundo sobe
    assert down[4:, 1].mean() < 360 - 20


def test_data_advice():
    from perpetual_edit.training import data_advice
    assert data_advice({"normal": 300, "beber": 320}, {"normal": 3, "beber": 2}) == []
    tips = data_advice({"normal": 274, "beber_agua": 566}, {"normal": 1, "beber_agua": 1})
    assert len(tips) == 2 and "normal, beber_agua" in tips[0] and "equilibre" in tips[1], tips


def test_recorder():
    import subprocess

    from perpetual_edit import recorder as rec_mod
    from perpetual_edit.audio import FFMPEG, AudioBus

    rec_mod.REC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "test_rec")
    bus = AudioBus()
    msgs = []
    rec = rec_mod.SessionRecorder(bus, on_done=msgs.append)
    rec.start(320, 240)
    frame = np.zeros((240, 320, 3), np.uint8)
    t0 = time.perf_counter()
    played = False
    while time.perf_counter() - t0 < 2.0:
        if not played and time.perf_counter() - t0 > 0.5:
            cfg, _ = load_config()
            bus.play(cfg["pre_edit_sound"], 0.0)
            played = True
        rec.write(frame)
        time.sleep(1 / 20)  # câmera "lenta": o gravador deve compensar
    rec.stop()
    rec.wait()
    bus.close()
    assert msgs and "salva" in msgs[0], msgs
    out = rec.out_path
    info = subprocess.run([FFMPEG, "-i", out], capture_output=True, text=True).stderr
    assert "Video: h264" in info and "Audio: aac" in info, info
    dur = [l for l in info.splitlines() if "Duration" in l][0]
    print(f"  recorder: {dur.strip()}")
    os.remove(out)


def test_player(cfg):
    from perpetual_edit.audio import AudioBus
    from perpetual_edit.player import EditPlayer, composite

    bus = AudioBus()
    player = EditPlayer(bus)
    # clipe de teste gerado por tools/make_placeholders.py (não está mais no config), trecho 0.5 -> 3.5
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    e = {"file": os.path.join(root, "videos", "edit_3.mp4"), "start": 0.5, "end": 3.5}
    player.load("edit_3", e)
    assert player.wav and os.path.exists(player.wav), "áudio não extraído"
    player.play(0.0)  # volume 0: só testa o relógio
    frames, t0 = 0, time.perf_counter()
    cam = np.zeros((720, 1280, 3), np.uint8)
    while True:
        fr = player.frame()
        if fr is None:
            break
        out = composite(cam, fr, "fullscreen")
        assert out.shape == cam.shape
        out = composite(cam, fr, "pip", cfg["pip"])
        assert out.shape == cam.shape
        frames += 1
        time.sleep(1 / 30)
    dur = time.perf_counter() - t0
    assert abs(dur - 3.0) < 0.3, f"duração {dur:.2f}s, esperado 3.0s"
    bus.close()
    print(f"  player: {frames} frames em {dur:.2f}s")


def test_detector():
    import cv2
    from perpetual_edit.detection import Detector

    det = Detector()
    img = np.full((720, 1280, 3), 90, np.uint8)
    t0 = time.perf_counter()
    for _ in range(20):
        f = det.process(img)
    ms = (time.perf_counter() - t0) / 20 * 1000
    assert not f.face_found
    det.close()
    print(f"  detector: {ms:.1f} ms/frame (face + gesture, CPU)")


if __name__ == "__main__":
    test_parse_time()
    cfg = test_config()
    test_triggers(cfg)
    test_phase2_evaluators()
    test_custom_trigger()
    print("ok: config + gatilhos (incl. desligar pelo menu)")
    test_popup()
    print("ok: pop-up")
    test_slot()
    print("ok: janela fixa (slot)")
    test_face_cube()
    print("ok: cubo 3D do rosto")
    test_data_advice()
    test_new_models()
    print("ok: criar modelos (mãos + recorte aberto)")
    test_recorder()
    print("ok: gravação")
    test_player(cfg)
    print("ok: player")
    if has_templates(cfg):
        test_template(cfg)
        test_auto_templates(cfg)
    else:
        test_demo_fallback(cfg)
        print("  sem mp4 em capcut_video/: testes de reprodução de template e do arquivo de replays pulados")
    test_importer_detect()
    print("ok: templates + importador")
    test_keyed()
    test_importer_green()
    print("ok: tela verde (composição + importador)")
    if has_templates(cfg):
        test_archive(cfg)
        print("ok: arquivo de replays")
    test_detector()
    print("ok: detector")
