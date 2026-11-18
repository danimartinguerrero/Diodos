#!/usr/bin/env python3
"""Recupera los puntos (V_D, I_D) de la figura vectorial Diodos_comparativa.pdf.

La figura original (matplotlib) se convierte a SVG con pdftocairo y se leen las
polilíneas de cada serie; los ejes van de -5 a 3 V y de -3 a 5 mA.
Uso: python3 scripts/extraer_datos.py <comparativa.svg>   ->  datos/*.dat
"""
import os
import re
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
X0, X1, V0, V1 = 30.441406, 436.675781, -5.0, 3.0      # caja de ejes (pt)
Y0, Y1, I0, I1 = 246.808594, 23.785156, -3.0, 5.0
SERIES = {
    "rgb(43.920898%, 43.920898%, 43.920898%)": "pn",
    "rgb(90.194702%, 38.430786%, 18.03894%)": "zener",
    "rgb(58.03833%, 25.489807%, 76.861572%)": "ir",
    "rgb(100%, 0%, 0%)": "rojo",
    "rgb(100%, 94.900513%, 0%)": "amarillo",
    "rgb(0%, 83.135986%, 2.745056%)": "verde",
    "rgb(0%, 31.764221%, 100%)": "azul",
    "rgb(32.940674%, 90.979004%, 76.469421%)": "blanco",
}

svg = open(sys.argv[1]).read()
os.makedirs(OUT, exist_ok=True)
for attrs in re.findall(r"<path ([^>]*)/>", svg):
    st = re.search(r'stroke="([^"]*)"', attrs)
    d = re.search(r' d="([^"]*)"', attrs)
    if not (st and d) or st.group(1) not in SERIES or " C " in d.group(1):
        continue
    nums = list(map(float, re.findall(r"-?\d+\.?\d*", d.group(1))))
    pts = list(zip(nums[0::2], nums[1::2]))
    if len(pts) < 50:               # descarta la muestra de la leyenda
        continue
    name = SERIES[st.group(1)]
    with open(os.path.join(OUT, f"{name}.dat"), "w") as f:
        f.write("VD ID\n")
        for x, y in pts:
            v = V0 + (x - X0) * (V1 - V0) / (X1 - X0)
            i = I0 + (y - Y0) * (I1 - I0) / (Y1 - Y0)
            f.write(f"{v:.4f} {i:.4f}\n")
    print(name, len(pts))
