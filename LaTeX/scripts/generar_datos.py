#!/usr/bin/env python3
"""Genera los datos ILUSTRATIVOS (simulados) de la guía de la Práctica 3.

Simula exactamente el experimento: el DAQ impone una rampa V_in en AO-0,
la corriente atraviesa R = 1 kOhm y el diodo, y AI-1 mide V_D (con ruido).
Cada diodo se modela con la ecuación de Shockley más una resistencia serie
y, para el Zener, una rama de ruptura inversa.

Uso:  python3 scripts/generar_datos.py      (desde la carpeta LaTeX/)
Salida: datos/*.dat  y  datos/tabla_umbral.tex
"""
import os
import numpy as np

rng = np.random.default_rng(2026)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
os.makedirs(OUT, exist_ok=True)

VT = 0.02585            # tensión térmica a 300 K (V)
R = 1000.0              # resistencia limitadora (Ohm)
V_START, V_END, N = -5.0, 5.0, 101   # rampa del DAQ
SIGMA_AI = 1.0e-3       # ruido de la entrada analógica (V)
H, C, E = 6.62607015e-34, 2.99792458e8, 1.602176634e-19


def make_diode(v_1mA, n, rs, vz=None, nz=1.0):
    """I(Vj) con Shockley ajustado para que I(v_1mA) ~ 1 mA."""
    i_s = 1e-3 / np.exp(v_1mA / (n * VT))

    def i_of_vj(vj):
        i = i_s * (np.exp(np.clip(vj / (n * VT), -200, 200)) - 1.0)
        if vz is not None:
            i -= 1e-3 * np.exp(np.clip((-vj - vz) / (nz * VT), -200, 200))
        return i
    return i_of_vj, rs


def solve(vin, diode):
    """Resuelve V_in = V_j + I(V_j) (R + R_s) por bisección."""
    i_of_vj, rs = diode
    lo, hi = -abs(vin) - 1.0, abs(vin) + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = mid + i_of_vj(mid) * (R + rs) - vin
        lo, hi = (mid, hi) if f < 0 else (lo, mid)
    vj = 0.5 * (lo + hi)
    i = i_of_vj(vj)
    return vj + i * rs


def measure(diode):
    vin = np.linspace(V_START, V_END, N)
    vd_true = np.array([solve(v, diode) for v in vin])
    vd = vd_true + rng.normal(0, SIGMA_AI, N)
    i_ma = (vin - vd) / (R / 1000.0)        # I (mA) = (V_in - V_D)/R[kOhm]
    return vin, vd, i_ma


def threshold(vd, i_ma):
    """Extrapolación lineal de la parte alta de la curva hasta I = 0."""
    imax = i_ma.max()
    m = i_ma > 0.5 * imax
    p = np.polyfit(vd[m], i_ma[m], 1)
    return -p[1] / p[0], p


def save(name, cols, header):
    np.savetxt(os.path.join(OUT, name), np.column_stack(cols),
               fmt="%.4e", header=header, comments="")


# ----------------------------------------------------------------- diodos
diodes = {
    "pn":     dict(d=make_diode(0.62, 1.9, 3.0)),
    "zener":  dict(d=make_diode(0.70, 1.6, 5.0, vz=3.3, nz=1.2)),
}
# LEDs: (lambda pico nm, V a 1 mA, n, Rs, color html, nombre, material)
leds = [
    ("ir",       940, 1.10, 1.9, 4.0, "IR",       "GaAs"),
    ("rojo",     630, 1.78, 2.0, 8.0, "Rojo",     "AlGaInP"),
    ("amarillo", 590, 1.87, 2.0, 8.0, "Amarillo", "AlGaInP"),
    ("verde",    525, 2.50, 2.6, 12.0, "Verde",   "InGaN"),
    ("azul",     470, 2.62, 2.6, 12.0, "Azul",    "InGaN"),
    ("blanco",   455, 2.66, 2.6, 12.0, "Blanco",  "InGaN + fósforo"),
    ("uv",       400, 3.02, 2.8, 15.0, "UV",      "InGaN"),
]
for key, lam, v1, n, rs, *_ in leds:
    diodes[key] = dict(d=make_diode(v1, n, rs), lam=lam)

rows = []
for key, item in diodes.items():
    vin, vd, i = measure(item["d"])
    save(f"iv_{key}.dat", (np.arange(N), vin, vd, i), "k Vin VD ID")
    vth, p = threshold(vd, i)
    item["vth"] = vth
    # recta de extrapolación (para dibujarla)
    xs = np.array([vth, vd.max()])
    save(f"fit_{key}.dat", (xs, np.polyval(p, xs)), "VD ID")
    print(f"{key:9s}  V_umbral = {vth:.3f} V   I_max = {i.max():.2f} mA")

# tensión Zener: extrapolación de la rama inversa hasta I = 0
vin, vd, i = measure(diodes["zener"]["d"])
m = i < 0.5 * i.min()
pz = np.polyfit(vd[m], i[m], 1)
vz = -pz[1] / pz[0]
xs = np.array([vd.min(), vz])
save("fitz_zener.dat", (xs, np.polyval(pz, xs)), "VD ID")
print(f"zener     V_Z = {vz:.3f} V")
# ------------------------------------------------- semilog del diodo pn
vin, vd, i = measure(diodes["pn"]["d"])
m = (i > 5e-3)
save("semilog_pn.dat", (vd[m], i[m]), "VD ID")
mf = (i > 0.02) & (i < 1.0)
ps = np.polyfit(vd[mf], np.log(i[mf]), 1)
n_fit, is_fit = 1.0 / (ps[0] * VT), np.exp(ps[1])      # I_S en mA
xs = np.linspace(0.30, 0.72, 50)
save("semilog_fit.dat", (xs, is_fit * np.exp(xs / (n_fit * VT))), "VD ID")
print(f"Shockley pn: n = {n_fit:.2f}, I_S = {is_fit*1e6:.2f} nA")

# ------------------------------------- recta de carga del LED rojo
rojo = diodes["rojo"]["d"]
pts = [(v, solve(v, rojo)) for v in (2.0, 3.0, 4.0, 5.0)]
save("recta_carga.dat", ([p[0] for p in pts], [p[1] for p in pts],
     [(p[0] - p[1]) for p in pts]), "Vin VD ID")
vv = np.linspace(1.50, 2.10, 200)
ii = np.array([rojo[0](v) * 1e3 for v in vv])   # sin R_s: curva del chip
ok = ii < 6.0
save("curva_rojo.dat", (vv[ok], ii[ok]), "VD ID")

# ------------------------------------------------- V_umbral frente a E_foton
lines, ex, ey = [], [], []
for key, lam, v1, n, rs, nombre, mat in leds:
    ef = 1239.84 / lam
    vth = diodes[key]["vth"]
    hest = E * vth * lam * 1e-9 / C
    lines.append(f"{nombre} & {mat} & {lam} & \\num{{{ef:.2f}}} & "
                 f"\\num{{{vth:.2f}}} & \\num{{{hest/1e-34:.2f}}} \\\\")
    if key != "blanco":
        ex.append(ef); ey.append(vth)
pf = np.polyfit(ex, ey, 1)
save("umbral_energia.dat", (np.array(ex), np.array(ey)), "E Vth")
xs = np.array([1.1, 3.4])
save("umbral_fit.dat", (xs, np.polyval(pf, xs)), "E Vth")
with open(os.path.join(OUT, "tabla_umbral.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(OUT, "ajuste_umbral.tex"), "w") as f:
    f.write(f"\\newcommand\\pendienteajuste{{{pf[0]:.2f}}}\n"
            f"\\newcommand\\ordenadaajuste{{{pf[1]:.2f}}}\n"
            f"\\newcommand\\umbralpn{{{diodes['pn']['vth']:.2f}}}\n"
            f"\\newcommand\\umbralzener{{{diodes['zener']['vth']:.2f}}}\n"
            f"\\newcommand\\tensionzener{{{-vz:.2f}}}\n"
            f"\\newcommand\\nshockley{{{n_fit:.2f}}}\n"
            f"\\newcommand\\isshockley{{{is_fit*1e6:.1f}}}\n")
print("ajuste V = a E + b :", pf)

# ------------------------------------------------- espectros de emisión
lam = np.linspace(350, 1000, 651)
g = lambda l0, fwhm: np.exp(-4 * np.log(2) * (lam - l0) ** 2 / fwhm ** 2)
spec = {"ir": g(940, 50), "rojo": g(630, 18), "amarillo": g(590, 16),
        "verde": g(525, 32), "azul": g(470, 24), "uv": g(400, 14),
        "blanco": np.maximum(g(455, 22), 0) + 0.55 * g(565, 120)}
spec["blanco"] /= spec["blanco"].max()
save("espectros.dat", [lam] + [spec[k] for k in
     ("ir", "rojo", "amarillo", "verde", "azul", "uv", "blanco")],
     "lam ir rojo amarillo verde azul uv blanco")
