"""Importa templates do CapCut: analisa o mp4 exportado e gera templates/<nome>.json.

Funciona melhor com o export de "mapa de espaços" (intro clip, clip 1, clip 2...):
cada corte vira um espaço preenchido com o seu replay. Flashes e fades são
detectados e recriados; a música vem do próprio vídeo. O vídeo em si nunca
aparece no edit (nada de marca d'água).

Uso:
  python tools\\import_template.py                       # todos os mp4 de capcut_video\\
  python tools\\import_template.py "capcut_video\\x.mp4"  # um arquivo
  python tools\\import_template.py --dry-run              # só mostra o que faria
  python tools\\import_template.py --force                # recria mesmo se já existir

Tela verde: vídeos em capcut_video\\green\\ (ou *_green.mp4, ou com --green) são o
template exportado com os espaços preenchidos por green_slots\\green_A/B.png. Viram
trechos "keyed": o vídeo aparece inteiro e só o verde é trocado pelo seu replay.
"""
import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perpetual_edit.config import ROOT  # noqa: E402

TEMPLATES_DIR = os.path.join(ROOT, "templates")
# Vídeos de capcut_video\ que não são templates (pulados na varredura automática;
# passando o caminho direto na linha de comando, ainda dá para importar).
SKIP_VIDEOS = {"inspiracao.mp4"}

# Janela do replay que o app grava: de BEFORE s antes do gatilho até LOADING s depois.
BEFORE, LOADING = 1.5, 2.0
# Momentos do gesto (s após o gatilho) usados em sequência nos cortes rápidos.
MOMENTS = [0.0, 0.9, 0.3, 1.2, 0.6, 1.4, -0.3, 1.0, 0.2, 0.8, 0.5, 1.3]
# Efeitos que alternam nos cortes secos, para não ficar repetitivo.
CUT_STYLES = [
    [{"type": "zoom", "from": 1.25, "to": 1.1, "anchor": "face"}],
    [{"type": "zoom", "from": 1.0, "to": 1.15, "anchor": "face"}],
    [{"type": "shake", "amount": 0.02, "rotate": 3}],
    [{"type": "zoom", "from": 1.3, "to": 1.2, "anchor": "face"}],
]


def analyze(path):
    """Brilho médio e mudança média por frame (num frame reduzido, rápido)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bright, diff, prev = [], [], None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        s = cv2.resize(f, (90, 120)).astype(np.float32)
        bright.append(float(s.mean()))
        diff.append(0.0 if prev is None else float(np.abs(s - prev).mean()))
        prev = s
    cap.release()
    return fps, (w, h), np.array(bright), np.array(diff)


def detect(fps, bright, diff):
    """Devolve (cortes [(frame, é_flash)], frame onde começa o preto final, fade_in em s)."""
    n = len(bright)
    thr = max(25.0, 4 * float(np.median(diff)))
    # Candidatos: mudança grande ou salto de brilho (flash). O apagar de um flash também
    # gera mudança grande, então entre candidatos próximos fica só o mais forte.
    cand = []
    for i in range(1, n):
        flash = bool(bright[i] - bright[i - 1] > 40)
        if diff[i] > thr or flash:
            cand.append((diff[i] + (1000 if flash else 0), i, flash))
    gap = int(0.2 * fps)
    cuts = []
    for _, i, flash in sorted(cand, reverse=True):
        if all(abs(i - j) >= gap for j, _ in cuts):
            cuts.append((i, flash))
    cuts.sort()
    # preto final: frames quase pretos até o fim
    black = n
    while black > 0 and bright[black - 1] < 1.0:
        black -= 1
    black = black if n - black >= int(0.2 * fps) else n
    # fade de entrada: começa escuro e sobe até o brilho típico do 1º trecho
    first_end = cuts[0][0] if cuts else black
    med = float(np.median(bright[:first_end])) if first_end else 0.0
    fade_in = 0.0
    if med > 15 and bright[0] < 0.3 * med:
        k = int(np.argmax(bright[:first_end] >= 0.9 * med))
        fade_in = round(k / fps, 2)
    return cuts, black, fade_in


def build(name, video_rel, fps, size, bright, diff):
    n = len(bright)
    cuts, black, fade_in = detect(fps, bright, diff)
    bounds = [0] + [c for c, _ in cuts if c < black] + [black]
    flashes = {c for c, f in cuts if f}
    segments, k_moment, k_cut = [], 0, 0
    for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
        start, end = round(a / fps, 2), round(b / fps, 2)
        dur = end - start
        seg = {"start": start, "end": end, "source": "replay"}
        effects = []
        if dur >= 2.5:  # espaço longo (intro): o replay inteiro
            seg["replay_at"] = -BEFORE
            seg["speed"] = round(min(1.0, (BEFORE + LOADING - 0.1) / dur), 2)
        elif dur >= 1.0:  # médio: o gesto em câmera lenta
            seg["replay_at"], seg["speed"] = -0.5, 0.7
        else:  # corte rápido: um momento diferente do gesto a cada espaço
            speed = 0.7 if dur >= 0.7 else 1.0
            at = MOMENTS[k_moment % len(MOMENTS)]
            k_moment += 1
            seg["replay_at"] = round(float(np.clip(at, -BEFORE, LOADING - dur * speed - 0.05)), 2)
            if speed != 1.0:
                seg["speed"] = speed
        if i == 0 and fade_in:
            effects.append({"type": "fade_in", "duration": fade_in})
        if a in flashes:
            effects += [{"type": "flash"}, {"type": "vignette", "strength": 0.85}]
        elif i > 0 and dur < 2.5:
            effects += CUT_STYLES[k_cut % len(CUT_STYLES)]
            k_cut += 1
        if b == black and black < n:  # último espaço antes do preto final: fade out
            effects.append({"type": "fade_out", "duration": round(min(0.5, dur), 2)})
        if effects:
            seg["effects"] = effects
        segments.append(seg)
    if black < n:
        segments.append({"start": round(black / fps, 2), "end": round(n / fps, 2), "source": "color", "color": [0, 0, 0]})
    w, h = size
    out_w = 540
    return {
        "name": name,
        "video": video_rel,
        "audio": "video",
        "size": [out_w, int(round(out_w * h / w / 2)) * 2],
        "duration": round(n / fps, 2),
        "replay": {"before_s": BEFORE},
        "imported": {"cuts": len(cuts), "flashes": len(flashes), "fade_in": fade_in,
                     "black_tail_s": round((n - black) / fps, 2)},
        "segments": segments,
    }


# ---------------------------------------------------------------- tela verde
GREEN_MIN = 0.02     # fração mínima da tela verde para o quadro contar como "espaço"
HUE_JUMP = 3.5       # salto de tom entre quadros seguidos que indica troca de slot sem pausa
SAME_HUE = 1.0       # tons até essa distância são o mesmo slot (o filtro do template deixa
                     # os slots a só ~2-4 de distância, então a tolerância é pequena)
MERGE_GAP_S = 0.2    # verdes do mesmo tom separados por menos que isso = o mesmo espaço (flash)
MIN_RUN_S = 0.12     # trechos verdes menores que isso são transição: juntam ao vizinho


def analyze_green(path):
    """Para cada quadro: fração verde e tom mediano do verde (-1 se não houver)."""
    from perpetual_edit.template import key_mask
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cover, hue = [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        m, hsv = key_mask(cv2.resize(f, (180, max(1, round(180 * h / w)))))
        sel = m > 0
        cover.append(float(sel.mean()))
        hue.append(float(np.median(hsv[:, :, 0][sel])) if sel.sum() > 20 else -1.0)
    cap.release()
    return fps, (w, h), np.array(cover), np.array(hue)


def green_runs(fps, cover, hue):
    """Trechos contínuos de verde do mesmo tom: [(início, fim exclusivo, tom)] em quadros."""
    n = len(cover)
    green = cover >= GREEN_MIN
    min_len = max(1, int(MIN_RUN_S * fps))
    runs = []
    for i in range(n):
        if not green[i]:
            continue
        # Corta quando o verde some ou quando o tom salta e fica no valor novo (troca de slot
        # sem pausa). Num fade o tom só "escorrega" alguns pontos por quadro: não corta.
        jump = (i > 0 and green[i - 1] and abs(hue[i] - hue[i - 1]) > HUE_JUMP
                and all(green[j] and abs(hue[j] - hue[i]) <= SAME_HUE for j in range(i, min(n, i + 3))))
        # degrau: tom estável antes, estável depois e diferente (corte direto entre slots
        # com tons vizinhos, ex.: 56 -> 58)
        if not jump and 3 <= i <= n - 3 and green[i - 3:i + 3].all():
            prev, nxt = hue[i - 3:i], hue[i:i + 3]
            jump = (np.ptp(prev) <= SAME_HUE and np.ptp(nxt) <= SAME_HUE
                    and abs(np.median(nxt) - np.median(prev)) >= 1.5)
        if runs and runs[-1][1] == i and not jump:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1, 0.0])
    for r in runs:
        r[2] = float(np.median(hue[r[0]:r[1]]))
    # transições de 1-3 quadros: juntam ao vizinho colado mais longo
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for k, r in enumerate(runs):
            if r[1] - r[0] >= min_len:
                continue
            nbrs = [j for j in (k - 1, k + 1) if 0 <= j < len(runs) and abs(runs[j][1 if j < k else 0] - r[0 if j < k else 1]) <= 1]
            if not nbrs:
                continue
            j = max(nbrs, key=lambda j: runs[j][1] - runs[j][0])
            keep = runs[j]
            keep[0], keep[1] = min(keep[0], r[0]), max(keep[1], r[1])
            del runs[k]
            changed = True
            break
    # mesmo tom separado por um piscar curto (flash): é o mesmo espaço
    merged = []
    for r in runs:
        if merged and abs(r[2] - merged[-1][2]) <= SAME_HUE and r[0] - merged[-1][1] < MERGE_GAP_S * fps:
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))
    return [(a, b, round(h, 1)) for a, b, h in merged]


def build_green(name, video_rel, fps, size, cover, hue):
    """Template de tela verde: um trecho "keyed" por espaço. Cada trecho vai do início do
    seu verde até o início do próximo (o que não é verde mostra o vídeo original)."""
    n = len(cover)
    runs = green_runs(fps, cover, hue)
    segments, k_moment = [], 0
    for idx, (a, b, h) in enumerate(runs):
        start = 0.0 if idx == 0 else round(a / fps, 2)
        end = round((runs[idx + 1][0] if idx + 1 < len(runs) else n) / fps, 2)
        green_dur = (b - a) / fps
        lead = a / fps - start  # tempo entre o início do trecho e o verde aparecer
        seg = {"start": start, "end": end, "source": "keyed", "slot_hue": h}
        if idx == 0 or green_dur >= 2.5:  # 1º espaço ou espaço longo: o replay inteiro
            speed = round(min(1.0, (BEFORE + LOADING - 0.1) / max(green_dur, 1e-3)), 2)
            seg["replay_at"] = round(-BEFORE - lead * speed, 2)
            if speed != 1.0:
                seg["speed"] = speed
        elif green_dur >= 1.0:  # médio: o gesto em câmera lenta
            seg["replay_at"], seg["speed"] = round(-0.5 - lead * 0.7, 2), 0.7
        else:
            # corte rápido: cada espaço mostra um momento diferente do gesto
            at = MOMENTS[k_moment % len(MOMENTS)]
            k_moment += 1
            seg["replay_at"] = round(float(np.clip(at - lead, -BEFORE, LOADING - green_dur - 0.05)), 2)
        segments.append(seg)
    w, h_px = size
    out_w = 540
    return {
        "name": name,
        "video": video_rel,
        "audio": "video",
        "size": [out_w, int(round(out_w * h_px / w / 2)) * 2],
        "duration": round(n / fps, 2),
        "replay": {"before_s": BEFORE},
        "key": {"ref_value": slots_ref_value()},
        "imported": {"mode": "green", "spaces": len(runs)},
        "segments": segments,
    }


def slots_ref_value():
    """Brilho do verde dos slots (green_slots/slots.json), usado para medir fades e flashes."""
    try:
        with open(os.path.join(ROOT, "green_slots", "slots.json"), "r", encoding="utf-8") as f:
            return json.load(f)["value"]
    except (OSError, KeyError, json.JSONDecodeError):
        return 230


def is_green_video(path):
    parts = os.path.normcase(os.path.abspath(path)).split(os.sep)
    return "green" in parts[-2:-1] or os.path.splitext(os.path.basename(path))[0].lower().endswith("_green")


def slug(path, green=False):
    stem = os.path.splitext(os.path.basename(path))[0]
    s = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    if green:  # templates de tela verde usam o nome que você deu: "lossing_green.mp4" -> "lossing"
        s = re.sub(r"_?green$", "", s) or "template"
        return s
    return f"capcut_{s or 'template'}"


def existing_templates():
    out = []
    for p in glob.glob(os.path.join(TEMPLATES_DIR, "*.json")):
        try:
            with open(p, "r", encoding="utf-8-sig") as f:  # -sig: aceita BOM
                out.append((p, json.load(f)))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"   aviso: não consegui ler {os.path.basename(p)} ({exc})")
    return out


def same_cuts(spec, duration, cut_times):
    """True se um template existente tem a mesma duração e os mesmos cortes (reexportação)."""
    if abs(spec.get("duration", -1) - duration) > 0.1 or not cut_times:
        return False
    starts = [s["start"] for s in spec.get("segments", [])]
    hits = sum(any(abs(c - s) <= 0.05 for s in starts) for c in cut_times)
    return hits >= 0.8 * len(cut_times)


def free_name(base, out_dir):
    name, suffix = base, iter("bcdefghijklmnopqrstuvwxyz")
    while os.path.exists(os.path.join(out_dir, f"{name}.json")):
        name = f"{base}_{next(suffix)}"
    return name


def import_green(path, rel, owner, args, existing):
    fps, size, cover, hue = analyze_green(path)
    if owner:
        out, name = owner, os.path.splitext(os.path.basename(owner))[0]
    else:
        name = free_name(slug(path, green=True), args.out)
        out = os.path.join(args.out, f"{name}.json")
    spec = build_green(name, rel, fps, size, cover, hue)
    info = spec["imported"]
    print(f"   tela verde: {spec['duration']}s, {info['spaces']} espaços, "
          f"verde em {100 * float((cover >= GREEN_MIN).mean()):.0f}% do vídeo")
    for s in spec["segments"]:
        print(f"     {s['start']:6.2f}-{s['end']:6.2f}s  tom {s['slot_hue']:5.1f}  replay_at {s['replay_at']:+.2f}"
              + (f"  speed {s['speed']}" if "speed" in s else ""))
    if not spec["segments"]:
        print("   aviso: nenhum verde encontrado. Os espaços foram preenchidos com green_slots/green_A/B.png?")
        return
    if args.dry_run:
        print(f"   (dry-run) geraria {os.path.relpath(out, ROOT)}")
        return
    os.makedirs(args.out, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    existing.append((out, spec))
    print(f"   ok: {os.path.relpath(out, ROOT)}")


def main():
    ap = argparse.ArgumentParser(description="Importa templates do CapCut (mp4) para templates/*.json")
    ap.add_argument("videos", nargs="*", help="arquivos mp4 (padrão: capcut_video/*.mp4)")
    ap.add_argument("--force", action="store_true", help="recria mesmo se já existir template para o vídeo")
    ap.add_argument("--dry-run", action="store_true", help="só mostra o que seria feito")
    ap.add_argument("--out", default=TEMPLATES_DIR, help="pasta de saída dos .json")
    ap.add_argument("--green", action="store_true",
                    help="trata os vídeos como tela verde (automático para capcut_video/green/ e *_green.mp4)")
    args = ap.parse_args()

    videos = args.videos or [
        p for p in (sorted(glob.glob(os.path.join(ROOT, "capcut_video", "*.mp4")))
                    + sorted(glob.glob(os.path.join(ROOT, "capcut_video", "green", "*.mp4"))))
        if os.path.basename(p).lower() not in SKIP_VIDEOS]
    if not videos:
        print("nenhum mp4 encontrado em capcut_video/")
        return
    existing = existing_templates()
    for path in videos:
        path = os.path.abspath(path)
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        print(f"\n== {rel}")
        owner = next((p for p, s in existing if os.path.normcase(os.path.abspath(
            os.path.join(ROOT, s.get("video", "")))) == os.path.normcase(path)), None)
        owner_spec = next((s for p, s in existing if p == owner), {})
        if owner and owner_spec.get("hand_tuned"):
            print(f"   já tem template ajustado à mão: {os.path.basename(owner)} (não recrio nem com --force)")
            continue
        if owner and not args.force:
            print(f"   já tem template: {os.path.basename(owner)} (use --force para recriar)")
            continue
        if args.green or is_green_video(path):
            import_green(path, rel, owner, args, existing)
            continue
        fps, size, bright, diff = analyze(path)
        cuts, black, _ = detect(fps, bright, diff)
        cut_times = [c / fps for c, _ in cuts]
        twin = next((p for p, s in existing if same_cuts(s, len(bright) / fps, cut_times)), None)
        if twin and not args.force:
            print(f"   mesmos cortes de {os.path.basename(twin)}: parece reexportação do mesmo template, pulando")
            continue
        # Nunca sobrescreve o template de OUTRO vídeo: se o nome já existe, escolhe um livre.
        # (--force só recria o template do mesmo vídeo, no mesmo arquivo.)
        if owner:
            out = owner
            name = os.path.splitext(os.path.basename(owner))[0]
        else:
            base, name = slug(path), slug(path)
            suffix = iter("bcdefghijklmnopqrstuvwxyz")
            while os.path.exists(os.path.join(args.out, f"{name}.json")):
                name = f"{base}_{next(suffix)}"
            out = os.path.join(args.out, f"{name}.json")
        spec = build(name, rel, fps, size, bright, diff)
        info = spec["imported"]
        print(f"   {spec['duration']}s, {len(spec['segments'])} trechos, {info['cuts']} cortes "
              f"({info['flashes']} flashes), fade de entrada {info['fade_in']}s, preto final {info['black_tail_s']}s")
        if len(spec["segments"]) <= 2:
            print("   aviso: quase nenhum corte detectado; pode ser uma prévia com filmagem de exemplo "
                  "(nesse caso vale ajustar o JSON à mão)")
        if args.dry_run:
            print(f"   (dry-run) geraria {os.path.relpath(out, ROOT)}")
            continue
        os.makedirs(args.out, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
        existing.append((out, spec))
        print(f"   ok: {os.path.relpath(out, ROOT)}")
    print("\nAperte R no app para carregar os templates novos.")


if __name__ == "__main__":
    main()
