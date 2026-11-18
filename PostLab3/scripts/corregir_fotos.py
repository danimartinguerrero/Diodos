#!/usr/bin/env python3
"""Corrige la perspectiva y mejora las fotos de pantalla de LabVIEW.

Uso: python3 scripts/corregir_fotos.py <foto_diagrama> <foto_panel>
Genera img/diagrama_bloques.jpg e img/panel_frontal.jpg.
"""
import os
import sys
import cv2
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "img")


def rectificar(img, src, w, h, origen, lienzo, recorte):
    """Homografía que lleva el cuadrilátero `src` a un rectángulo w x h."""
    x0, y0 = origen
    dst = np.float32([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]])
    m = cv2.getPerspectiveTransform(np.float32(src), dst)
    out = cv2.warpPerspective(img, m, lienzo, flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    x1, y1, x2, y2 = recorte
    return out[y1:y2, x1:x2]


def mejorar(img):
    # atenúa el muaré de fotografiar la pantalla y el ruido
    img = cv2.bilateralFilter(img, 7, 40, 7)
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    # balance de blancos (mundo gris sobre los tonos claros)
    f = img.astype(np.float32)
    claro = f.mean(axis=2) > np.percentile(f.mean(axis=2), 60)
    medias = f[claro].mean(axis=0)
    f *= medias.mean() / medias
    img = np.clip(f, 0, 255).astype(np.uint8)
    # contraste local (CLAHE sobre la luminancia)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    # estiramiento de niveles y máscara de enfoque
    lo, hi = np.percentile(img, (1, 99.5))
    img = np.clip((img.astype(np.float32) - lo) * 255 / (hi - lo), 0, 255)
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = np.clip(cv2.addWeighted(img, 1.6, blur, -0.6, 0), 0, 255).astype(np.uint8)
    return img


def main(diagrama, panel):
    os.makedirs(OUT, exist_ok=True)
    bd = cv2.imread(diagrama)
    # esquinas del marco del bucle For (rectángulo en la pantalla)
    src = [[377, 219], [1381, 255], [1356, 990], [392, 998]]
    bd = rectificar(bd, src, 985, 757, (400, 230), (2100, 1250), (60, 190, 1990, 1060))
    cv2.imwrite(os.path.join(OUT, "diagrama_bloques.jpg"), mejorar(bd),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    fp = cv2.imread(panel)
    # esquinas del área de trazado del XY Graph
    src = [[1360, 647], [1757, 657], [1718, 1032], [1337, 1029]]
    fp = rectificar(fp, src, 389, 378, (1360, 650), (2050, 1450), (270, 390, 1830, 1200))
    cv2.imwrite(os.path.join(OUT, "panel_frontal.jpg"), mejorar(fp),
                [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    main(*sys.argv[1:3])
