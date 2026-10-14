"""Safe Air Pedregal - Análisis y gráficos de dispersión (ENS160 + ThingSpeak)

Descarga las lecturas del canal público de ThingSpeak, calcula la relación
entre COV (TVOC) y eCO2 y genera:
  resultados/dispersion_cov_eco2.png   COV vs eCO2 (color = AQI) + recta de regresión
  resultados/dispersion_aqi_tiempo.png AQI vs tiempo
  resultados/datos.csv                 datos usados
  resultados/resumen.md                estadísticas del análisis
"""
import os
import sys
import json
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHANNEL_ID = os.environ.get("CHANNEL_ID", "3399651")
RESULTADOS = int(os.environ.get("RESULTADOS", "8000"))
SALIDA = "resultados"
COLORES = {1: "#22c55e", 2: "#facc15", 3: "#fb923c", 4: "#ef4444", 5: "#a855f7"}
NOMBRES = {1: "1 Excelente", 2: "2 Bueno", 3: "3 Moderado", 4: "4 Deficiente", 5: "5 Insalubre"}


def descargar():
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?results={RESULTADOS}"
    with urllib.request.urlopen(url, timeout=30) as r:
        feeds = json.load(r).get("feeds", [])
    df = pd.DataFrame(feeds)
    if df.empty:
        return df
    df = df.rename(columns={"field1": "AQI", "field2": "COV_ppb", "field3": "eCO2_ppm"})
    for c in ("AQI", "COV_ppb", "eCO2_ppm"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["fecha"] = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert("America/Bogota")
    df = df[(df["AQI"] >= 1) & (df["AQI"] <= 5)]
    return df[["fecha", "AQI", "COV_ppb", "eCO2_ppm"]].reset_index(drop=True)


def grafico_cov_eco2(d):
    x, y = d["COV_ppb"].values, d["eCO2_ppm"].values
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for aqi, color in COLORES.items():
        s = d[d["AQI"] == aqi]
        if len(s):
            ax.scatter(s["COV_ppb"], s["eCO2_ppm"], c=color, s=22, alpha=0.85,
                       label=NOMBRES[aqi], edgecolors="none")
    stats = {}
    if len(d) >= 3 and np.std(x) > 0 and np.std(y) > 0:
        m, b = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, color="black", lw=1.6, ls="--",
                label=f"Regresión: y = {m:.2f}x + {b:.1f}  (R² = {r**2:.2f})")
        stats = {"pendiente": m, "intercepto": b, "pearson": r, "spearman": rho, "r2": r**2}
    ax.set_xlabel("COV / TVOC (ppb)")
    ax.set_ylabel("eCO2 (ppm)")
    ax.set_title("Safe Air Pedregal - COV vs eCO2 (ENS160)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{SALIDA}/dispersion_cov_eco2.png")
    plt.close(fig)
    return stats


def grafico_aqi_tiempo(d):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.scatter(d["fecha"], d["AQI"], c=[COLORES[int(a)] for a in d["AQI"]], s=22, edgecolors="none")
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0.5, 5.5)
    ax.set_xlabel("Fecha y hora (Colombia)")
    ax.set_ylabel("AQI")
    ax.set_title("Safe Air Pedregal - AQI en el tiempo")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{SALIDA}/dispersion_aqi_tiempo.png")
    plt.close(fig)


def interpretar(r):
    a = abs(r)
    fuerza = "muy fuerte" if a >= 0.9 else "fuerte" if a >= 0.7 else "moderada" if a >= 0.4 else "débil" if a >= 0.2 else "muy débil o nula"
    signo = "positiva" if r > 0 else "negativa"
    return f"correlación {fuerza} {signo}"


def main():
    os.makedirs(SALIDA, exist_ok=True)
    df = descargar()
    if df.empty:
        print("ThingSpeak no devolvió datos.")
        sys.exit(1)

    df.to_csv(f"{SALIDA}/datos.csv", index=False)
    grafico_aqi_tiempo(df)

    cv = df.dropna(subset=["COV_ppb", "eCO2_ppm"])
    lineas = [
        "# Resumen del análisis - Safe Air Pedregal",
        "",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')} (hora del servidor)",
        f"Canal ThingSpeak: {CHANNEL_ID}",
        f"Lecturas con AQI: {len(df)}  |  Lecturas con COV y eCO2: {len(cv)}",
        f"Periodo: {df['fecha'].min():%Y-%m-%d %H:%M} a {df['fecha'].max():%Y-%m-%d %H:%M}",
        "",
        "## Distribución del AQI",
        "",
        "| AQI | Lecturas | % |",
        "|---|---|---|",
    ]
    for aqi in range(1, 6):
        n = int((df["AQI"] == aqi).sum())
        lineas.append(f"| {NOMBRES[aqi]} | {n} | {100*n/len(df):.1f} |")

    lineas += ["", "## COV vs eCO2", ""]
    if len(cv) >= 3:
        stats = grafico_cov_eco2(cv)
        if stats:
            lineas += [
                f"- Correlación de Pearson (r): {stats['pearson']:.3f}",
                f"- Correlación de Spearman: {stats['spearman']:.3f}",
                f"- Regresión lineal: eCO2 = {stats['pendiente']:.3f} x COV + {stats['intercepto']:.1f}",
                f"- R²: {stats['r2']:.3f}",
                f"- Interpretación: {interpretar(stats['pearson'])} entre COV y eCO2.",
                f"- COV: media {cv['COV_ppb'].mean():.1f} ppb, rango {cv['COV_ppb'].min():.0f} a {cv['COV_ppb'].max():.0f}",
                f"- eCO2: media {cv['eCO2_ppm'].mean():.1f} ppm, rango {cv['eCO2_ppm'].min():.0f} a {cv['eCO2_ppm'].max():.0f}",
            ]
        else:
            lineas.append("Los datos no tienen variación suficiente para calcular la regresión.")
    else:
        lineas.append("Aún no hay datos de COV y eCO2 en ThingSpeak (field2 y field3). "
                      "El firmware debe enviarlos para generar este gráfico.")

    with open(f"{SALIDA}/resumen.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
