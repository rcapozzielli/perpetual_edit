"""Carregamento e validação do config.json."""
import copy
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "camera_index": 0,
    "camera_backend": "msmf" if sys.platform == "win32" else "any",
    "camera_width": 1280,
    "camera_height": 720,
    "mirror": True,
    "detect_width": 640,
    "overlay_mode": "slot",
    "pip": {"scale": 0.38, "corner": "bottom_right", "margin": 24},
    # popup: altura do card (fração da tela), inclinação máxima sorteada, borda e margem em px
    "popup": {"height": 0.55, "tilt_deg": 4, "border": 6, "margin": 30},
    # slot: janela fixa no canto; width/big_width = fração da largura, aspect = largura/altura
    "slot": {"corner": "top_right", "width": 0.28, "big_width": 0.55, "big_chance": 0.33,
             "aspect": 1.4, "margin": 40, "top": 72},
    "global_cooldown_s": 4.0,
    "smoothing": 0.5,
    "pre_edit_sound": "sounds/pre.wav",
    "post_edit_sound": "sounds/post.wav",
    "pre_edit_duration": 0.6,
    "volume": {"edit": 1.0, "sfx": 0.8},
    "label_duration_s": 1.5,
    "custom_models": {},
    "auto_templates": True,
    "exclude_templates": [],  # templates/*.json que NÃO entram no sorteio (o arquivo continua lá)
    "archive": {"enabled": True, "dir": "replays", "max_clips": 30, "mix": 0.25, "min_face_ratio": 0.6,
                "clear_on_exit": True},
    "template_defaults": {"pre_edit_duration": 2.0, "pre_edit_sound": "sounds/loading.wav"},
    "triggers": {},
    "edits": {},
}

TRIGGER_DEFAULTS = {
    "enabled": True,
    "threshold": 0.5,
    "hold_frames": 8,
    "cooldown_s": 6.0,
    "edits": [],
}


def resolve(path):
    """Caminhos do config são relativos à raiz do projeto."""
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def parse_time(value):
    """Aceita número (segundos), "ss.xx", "mm:ss.xx" ou "hh:mm:ss.xx"."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    seconds = 0.0
    for part in str(value).strip().split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None, user_models=None):
    """Lê o config, completa com defaults e devolve (config, avisos).
    user_models: caminho do custom/models.json (trocado só nos testes)."""
    path = path or os.path.join(ROOT, "config.json")
    with open(path, "r", encoding="utf-8-sig") as f:  # -sig: aceita arquivo salvo com BOM (Bloco de Notas)
        raw = json.load(f)

    cfg = _merge(DEFAULTS, raw)
    warnings = []

    # Modelos criados pelo app (tecla N no modo treino) ficam em custom/models.json,
    # cada um com o seu gatilho. O config.json tem prioridade se usar o mesmo nome.
    user_models = user_models or os.path.join(ROOT, "custom", "models.json")
    if os.path.exists(user_models):
        try:
            with open(user_models, "r", encoding="utf-8-sig") as f:
                user = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"custom/models.json inválido ({exc})")
            user = {}
        for name, m in user.items():
            m = dict(m)
            trig = m.pop("trigger", None)
            cfg["custom_models"].setdefault(name, m)
            if trig and name not in cfg["triggers"]:
                cfg["triggers"][name] = {**trig, "type": "custom", "model": name}

    # Todo templates/*.json vira um edit automaticamente (nome = nome do arquivo),
    # a menos que já esteja declarado em "edits".
    if cfg["auto_templates"]:
        declared = {os.path.normcase(resolve(e["template"])) for e in cfg["edits"].values() if "template" in e}
        for p in sorted(glob.glob(os.path.join(ROOT, "templates", "*.json"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem in cfg["exclude_templates"]:
                continue
            if os.path.normcase(p) not in declared and stem not in cfg["edits"]:
                cfg["edits"][stem] = {"template": os.path.relpath(p, ROOT)}

    edits = {}
    for name, e in cfg["edits"].items():
        if "template" in e:
            e = {**cfg["template_defaults"], **e}  # padrões comuns a todos os templates
        # Aviso de "loading" antes do edit: som + barra na tela. Pode ser ajustado por edit.
        pre_edit = {
            "enabled": e.get("pre_edit", True),
            "duration": float(e.get("pre_edit_duration", cfg["pre_edit_duration"])),
            "sound": resolve(e.get("pre_edit_sound", cfg["pre_edit_sound"])),
        }
        if pre_edit["sound"] and not os.path.exists(pre_edit["sound"]):
            warnings.append(f"edit '{name}': pre_edit_sound não encontrado ({e.get('pre_edit_sound')})")
            pre_edit["sound"] = None
        if "template" in e:
            from .template import load_template  # import tardio: só quem usa template paga o custo
            try:
                spec = load_template(resolve(e["template"]))
            except Exception as exc:
                warnings.append(f"edit '{name}': template inválido ({exc})")
                continue
            edits[name] = {"template": resolve(e["template"]), "spec": spec, "pre_edit": pre_edit}
            continue
        file = resolve(e.get("file"))
        if not file or not os.path.exists(file):
            warnings.append(f"edit '{name}': arquivo não encontrado ({e.get('file')})")
            continue
        start = parse_time(e.get("start")) or 0.0
        end = parse_time(e.get("end"))
        if end is not None and end <= start:
            warnings.append(f"edit '{name}': end <= start, usando o clipe até o fim")
            end = None
        edits[name] = {"file": file, "start": start, "end": end, "pre_edit": pre_edit}

    # Clone limpo: os mp4 do CapCut não vão para o repositório, então nenhum template carrega.
    # Nesse caso os clipes de teste de videos/ (tools/make_placeholders.py) entram no lugar.
    if cfg["auto_templates"] and not any("template" in e for e in edits.values()):
        sound = resolve(cfg["pre_edit_sound"])
        pre_edit = {"enabled": True, "duration": float(cfg["pre_edit_duration"]),
                    "sound": sound if sound and os.path.exists(sound) else None}
        demos = sorted(glob.glob(os.path.join(ROOT, "videos", "edit_*.mp4")))
        for p in demos:
            stem = os.path.splitext(os.path.basename(p))[0]
            edits.setdefault(stem, {"file": p, "start": 0.0, "end": None, "pre_edit": pre_edit})
        if demos:
            warnings.append("nenhum template disponível: usando os clipes de teste de videos/")
    cfg["edits"] = edits

    for name, m in list(cfg["custom_models"].items()):
        if not 2 <= len(m.get("classes", [])) <= 9:  # classes são escolhidas pelas teclas 1-9
            warnings.append(f"custom_models '{name}': precisa de 2 a 9 classes")
            del cfg["custom_models"][name]

    triggers = {}
    for name, t in cfg["triggers"].items():
        t = _merge(TRIGGER_DEFAULTS, t)
        t.setdefault("type", name)
        t.setdefault("label", name.replace("_", " "))
        # Atalhos: "@templates" = todos os templates, "@clips" = todos os clipes comuns.
        # Sem nenhum template, "@templates" usa os clipes (ver o fallback acima).
        clips = [n for n, e in edits.items() if "template" not in e]
        groups = {"@templates": [n for n, e in edits.items() if "template" in e] or clips,
                  "@clips": clips}
        expanded = []
        for e in t["edits"]:
            for n in groups.get(e, [e]):
                if n not in expanded:
                    expanded.append(n)
        t["edits"] = expanded
        valid = [e for e in t["edits"] if e in edits]
        missing = [e for e in t["edits"] if e not in edits]
        if missing:
            warnings.append(f"gatilho '{name}': edits inexistentes {missing}")
        t["edits"] = valid
        if t["type"] == "custom":
            model = cfg["custom_models"].get(t.get("model"))
            if model is None:
                warnings.append(f"gatilho '{name}': modelo '{t.get('model')}' não está em custom_models")
            else:
                bad = [c for c in (t.get("from"), t.get("to")) if c is not None and c not in model["classes"]]
                if bad:
                    warnings.append(f"gatilho '{name}': classes {bad} não existem no modelo '{t['model']}'")
        triggers[name] = t
    cfg["triggers"] = triggers

    if cfg["overlay_mode"] not in ("slot", "popup", "fullscreen", "pip"):
        warnings.append(f"overlay_mode '{cfg['overlay_mode']}' inválido, usando slot")
        cfg["overlay_mode"] = "slot"

    for key in ("pre_edit_sound", "post_edit_sound"):
        p = resolve(cfg.get(key))
        if p and not os.path.exists(p):
            warnings.append(f"{key}: arquivo não encontrado ({cfg[key]})")
            p = None
        cfg[key] = p

    return cfg, warnings
