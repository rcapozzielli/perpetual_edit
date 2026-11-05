"""Gera as duas imagens verdes para montar templates de tela verde no CapCut.

  green_A.png: verde amarelado
  green_B.png: verde azulado

São lisas (sem número nem desenho): qualquer detalhe mais escuro viraria um
"buraco" na máscara quando o template escurece a tela. No CapCut, alterne:
Photo 1 = A, Photo 2 = B, Photo 3 = A... Assim dois espaços seguidos sempre
têm tons diferentes e o importador acha o corte entre eles, mesmo sem pausa.

A cor só marca ONDE você aparece. A imagem que entra em cada espaço vem do
seu replay (um momento diferente do gesto em cada espaço).

Uso:
  python tools\\make_green_slots.py     # gera green_slots\\green_A.png e green_B.png
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perpetual_edit.config import ROOT  # noqa: E402

# Tons do OpenCV (hue 0-180; verde puro = 60). Bem distantes entre si para continuarem
# separáveis mesmo quando o filtro do template "puxa" os tons para perto um do outro,
# e dentro da faixa de verde que o app troca (35-88).
HUES = {"A": 45, "B": 75}
SIZE = (1080, 1920)  # 9:16; o CapCut recorta o centro para 3:4 ou 1:1
VALUE = 230          # brilho (abaixo de 255 para sobrar espaço se o CapCut clarear)


def make_green(hue):
    w, h = SIZE
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[:, :] = (hue, 255, VALUE)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main():
    ap = argparse.ArgumentParser(description="Gera as imagens verdes A e B para templates de tela verde")
    ap.add_argument("--out", default=os.path.join(ROOT, "green_slots"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, hue in HUES.items():
        path = os.path.join(args.out, f"green_{name}.png")
        cv2.imwrite(path, make_green(hue))
        print(f"   {os.path.relpath(path, ROOT)}  (tom {hue})")
    # Referência para o importador: brilho original do verde (mede fades e flashes).
    with open(os.path.join(args.out, "slots.json"), "w", encoding="utf-8") as f:
        json.dump({"hues": HUES, "value": VALUE, "size": SIZE}, f, indent=2)
    print("ok. No CapCut: Photo 1 = A, Photo 2 = B, Photo 3 = A...")


if __name__ == "__main__":
    main()
