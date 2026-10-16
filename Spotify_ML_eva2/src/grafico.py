"""Funciones de graficos del proyecto (solo matplotlib).

Convencion de color en TODOS los graficos:
  naranjo = canciones con popularidad 0 / modelo o analisis CON ceros
  azul    = resto de canciones            / modelo o analisis SIN ceros
  verde   = modelo inicial (solo en el diagnostico de la linea horizontal)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

C_CERO = "#eb6834"
C_RESTO = "#2a78d6"
C_INICIAL = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "font.size": 11, "legend.frameon": False,
})


def _n(x):
    """Entero con punto de miles (estilo chileno): 80932 -> 80.932"""
    return f"{int(round(x)):,}".replace(",", ".")


def _d(x, nd=1):
    """Decimal con coma: 6.7 -> 6,7"""
    return f"{x:.{nd}f}".replace(".", ",")


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def mostrar(path, width=12):
    """Muestra en el notebook una imagen ya guardada (los graficos se guardan en output/Graficos)."""
    img = plt.imread(str(path))
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(width, width * h / w))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img); ax.axis("off")
    plt.show()


# --------------------------------------------------------------------------------------
# Funciones basicas (se mantienen por compatibilidad)
# --------------------------------------------------------------------------------------
def save_histogram(series, title, xlabel, ylabel, path, bins=20):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(series, bins=bins, color=C_RESTO)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


def save_barh(labels, values, title, xlabel, path):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels, values, color=C_RESTO)
    ax.set_title(title); ax.set_xlabel(xlabel)
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


# --------------------------------------------------------------------------------------
# EDA: ceros vs. resto
# --------------------------------------------------------------------------------------
def plot_dedup_pasos(report, path):
    """Barras apiladas: filas que quedan tras cada paso de deduplicacion, separando ceros y resto."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    labels = report["paso"].tolist()[::-1]
    ceros = report["ceros"].to_numpy()[::-1]
    resto = (report["filas"] - report["ceros"]).to_numpy()[::-1]
    pct = report["pct_ceros"].to_numpy()[::-1]
    y = np.arange(len(labels))
    ax.barh(y, resto, color=C_RESTO, edgecolor="white", linewidth=1.5, height=0.62, label="Popularidad > 0")
    ax.barh(y, ceros, left=resto, color=C_CERO, edgecolor="white", linewidth=1.5, height=0.62, label="Popularidad = 0")
    for yi, r, c, pc in zip(y, resto, ceros, pct):
        ax.text(r + c + 1500, yi, f"{_n(r + c)} filas  ·  {_d(pc, 1)}% ceros", va="center", color=INK, fontsize=10.5)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(0, report["filas"].max() * 1.45)
    ax.set_xlabel("Canciones (filas)")
    ax.xaxis.set_major_formatter(lambda v, _: _n(v))
    ax.grid(axis="y", visible=False)
    ax.set_title("Cada paso de limpieza elimina sobre todo canciones con popularidad 0")
    ax.legend(loc="lower right")
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


def plot_popularidad_ceros(y, path):
    """Histograma de popularidad: izquierda con ceros (barra 0 resaltada), derecha sin ceros."""
    y = pd.Series(y).astype(int)
    counts = y.value_counts().sort_index()
    n, z = len(y), int((y == 0).sum())
    rest = y[y > 0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    colors = [C_CERO if v == 0 else C_RESTO for v in counts.index]
    ax.bar(counts.index, counts.values, width=0.9, color=colors)
    ax.set_title(f"Con ceros  (n = {_n(n)})")
    ax.annotate(f"popularidad 0:\n{_n(z)} canciones ({_d(z / n * 100)}%)", xy=(0, z), xytext=(14, z * 0.93),
                color=INK, fontsize=10.5, arrowprops=dict(arrowstyle="-", color=INK2))
    ax.set_xlabel("Popularidad"); ax.set_ylabel("Canciones")
    ax.yaxis.set_major_formatter(lambda v, _: _n(v))
    ax = axes[1]
    rc = rest.value_counts().sort_index()
    ax.bar(rc.index, rc.values, width=0.9, color=C_RESTO)
    ax.axvline(rest.mean(), color=INK, linewidth=1.5, linestyle="--")
    ax.text(rest.mean() + 1.2, rc.max() * 0.97, f"media {_d(rest.mean())}\nmediana {_d(rest.median(), 0)}", va="top", color=INK, fontsize=10.5)
    ax.set_title(f"Sin ceros  (n = {_n(len(rest))})")
    ax.set_xlabel("Popularidad"); ax.set_ylabel("Canciones")
    ax.yaxis.set_major_formatter(lambda v, _: _n(v))
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


def _hist_pair(ax, zero, rest, col, bins=30):
    lo, hi = np.nanquantile(pd.concat([zero, rest]), [0.005, 0.995])
    edges = np.linspace(lo, hi, bins + 1)
    for s, color in ((rest, C_RESTO), (zero, C_CERO)):
        ax.hist(s.clip(lo, hi), bins=edges, density=True, color=color, alpha=0.30, histtype="stepfilled")
        ax.hist(s.clip(lo, hi), bins=edges, density=True, color=color, histtype="step", linewidth=1.8)
    ax.set_title(f"{col}")
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"media: ceros {_d(zero.mean(), 2)}  ·  resto {_d(rest.mean(), 2)}", fontsize=10, color=INK2)


def plot_audio_ceros(df, cols, path):
    """Distribucion de variables de audio: canciones con popularidad 0 vs. el resto (densidad)."""
    zero = df["es_cero"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), cols):
        _hist_pair(ax, df.loc[zero, col].astype(float), df.loc[~zero, col].astype(float), col)
    for ax in axes.ravel()[len(cols):]:
        ax.set_visible(False)
    fig.legend(handles=[Patch(color=C_CERO, alpha=0.6, label="Popularidad = 0"), Patch(color=C_RESTO, alpha=0.6, label="Popularidad > 0")],
               loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.02))
    fig.suptitle("Los ceros se superponen con el resto: solo son algo menos enérgicos y más acústicos", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); p = _save(fig, path); plt.close(fig); return p


def save_hist_ceros(df, col, title, path):
    """Una sola variable: densidad de ceros vs. resto."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    zero = df["es_cero"]
    _hist_pair(ax, df.loc[zero, col].astype(float), df.loc[~zero, col].astype(float), col)
    ax.set_title(title)
    ax.legend(handles=[Patch(color=C_CERO, alpha=0.6, label="Popularidad = 0"), Patch(color=C_RESTO, alpha=0.6, label="Popularidad > 0")], loc="upper left")
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


def plot_diferencia_estandarizada(smd, path):
    """Diferencia estandarizada de medias (ceros - resto) por variable de audio."""
    s = smd.sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(s.index, s.values, color=C_RESTO, height=0.6)
    ax.axvline(0, color=INK2, linewidth=1)
    for x in (-0.2, 0.2):
        ax.axvline(x, color=INK2, linewidth=1, linestyle=":")
    for i, v in enumerate(s.values):
        ax.text(v + (0.008 if v >= 0 else -0.008), i, _d(v, 2), va="center", ha="left" if v >= 0 else "right", fontsize=9.5, color=INK)
    lim = max(0.3, np.abs(s.values).max() * 1.35)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Diferencia estandarizada (ceros - resto)   ·   entre las líneas punteadas = efecto pequeño (|d| < 0,2)")
    ax.set_title(f"Diferencias de audio pequeñas: la mayor es |d| = {_d(np.abs(s.values).max(), 2)}")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


def plot_ceros_por_genero(gdf, overall_pct, path, top=15):
    """% de canciones con popularidad 0 en los generos con mayor proporcion de ceros."""
    g = gdf.sort_values("pct_ceros", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.barh(g.index, g["pct_ceros"], color=C_CERO, height=0.65)
    for i, (v, n) in enumerate(zip(g["pct_ceros"], g["n"])):
        ax.text(v + 0.4, i, f"{_d(v)}%  (n={_n(n)})", va="center", fontsize=9.5, color=INK)
    ax.axvline(overall_pct, color=INK, linestyle="--", linewidth=1.3, label=f"promedio general {_d(overall_pct)}%")
    ax.legend(loc="lower right")
    ax.set_xlim(0, g["pct_ceros"].max() * 1.3)
    ax.set_xlabel("% de canciones del género con popularidad 0")
    ax.set_title("Los ceros no se reparten parejo: se concentran en pocos géneros")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


def plot_popularidad_genero(gdf, path, top=15):
    """Popularidad media por genero, con ceros vs. sin ceros (los `top` generos segun la media sin ceros)."""
    g = gdf.sort_values("media_sin", ascending=False).head(top).iloc[::-1]
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(y + 0.19, g["media_con"], height=0.36, color=C_CERO, label="Con ceros")
    ax.barh(y - 0.19, g["media_sin"], height=0.36, color=C_RESTO, label="Sin ceros")
    ax.set_yticks(y); ax.set_yticklabels(g.index)
    ax.set_xlabel("Popularidad media")
    ax.set_title(f"{top} géneros con mayor popularidad media")
    ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


def plot_correlaciones(corr_con, corr_sin, path):
    """Correlacion lineal de cada variable con popularity: con ceros vs. sin ceros."""
    idx = corr_sin.abs().sort_values().index
    y = np.arange(len(idx))
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    ax.barh(y + 0.19, corr_con.reindex(idx), height=0.36, color=C_CERO, label="Con ceros")
    ax.barh(y - 0.19, corr_sin.reindex(idx), height=0.36, color=C_RESTO, label="Sin ceros")
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(idx)
    ax.set_xlabel("Correlación con popularity")
    ax.xaxis.set_major_formatter(lambda v, _: _d(v, 2))
    ax.set_title("Correlación con popularidad (ordenada por fuerza sin ceros)")
    ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


# --------------------------------------------------------------------------------------
# Modelos supervisados
# --------------------------------------------------------------------------------------
SCENARIOS = [
    ("Con ceros · test completo", C_CERO, None),
    ("Con ceros · test sin ceros", C_CERO, "//"),
    ("Sin ceros · test sin ceros", C_RESTO, None),
    ("Sin ceros · test completo", C_RESTO, "//"),
]


def plot_comparacion_modelos(res, path):
    """`res`: columnas Modelo, Escenario, MAE, RMSE, R2. Barras agrupadas por modelo."""
    models = list(dict.fromkeys(res["Modelo"]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, metric, title in zip(axes, ["MAE", "RMSE", "R2"], ["MAE (puntos, menor es mejor)", "RMSE (puntos, menor es mejor)", "R² (mayor es mejor)"]):
        w = 0.19
        for k, (name, color, hatch) in enumerate(SCENARIOS):
            vals = [res[(res["Modelo"] == m) & (res["Escenario"] == name)][metric].iloc[0] for m in models]
            xs = np.arange(len(models)) + (k - 1.5) * (w + 0.02)
            if hatch:
                ax.bar(xs, vals, w, facecolor="white", edgecolor=color, hatch=hatch, linewidth=1.5)
            else:
                ax.bar(xs, vals, w, color=color, edgecolor="white", linewidth=1.5)
            for x, v in zip(xs, vals):
                ax.text(x, v, _d(v, 2 if metric == "R2" else 1), ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(np.arange(len(models))); ax.set_xticklabels([m.replace("Regresion", "Regresión") for m in models])
        ax.set_title(title); ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)
        ax.yaxis.set_major_formatter(lambda v, _: _d(v, 1))
    fig.legend(handles=[Patch(facecolor=c if not h else "white", edgecolor=c, hatch=h, label=n) for n, c, h in SCENARIOS],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


def _hexpanel(ax, y, p, title, band=False, metrics=None):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    hb = ax.hexbin(y, p, gridsize=42, extent=(-1, 101, -1, 101), mincnt=1, cmap="Blues", norm=LogNorm(), linewidths=0.15)
    ax.plot([0, 100], [0, 100], color=INK2, linestyle="--", linewidth=1.4)
    edges = np.arange(0, 101, 5)
    mids, means = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (y >= a) & (y < b + (1 if b == 100 else 0))
        if m.sum() >= 20:
            mids.append((a + b) / 2); means.append(p[m].mean())
    ax.plot(mids, means, color=INK, linewidth=2, marker="o", markersize=3.5)
    if band:
        ax.axvspan(-1.5, 1.5, color=C_CERO, alpha=0.25, linewidth=0)
        ax.text(2.5, 97, "ceros", color=C_CERO, fontsize=10, fontweight="bold", va="top")
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    ax.set_xlabel("Popularidad real"); ax.set_title(title)
    if metrics:
        ax.text(0.97, 0.05, metrics, transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=GRID))
    return hb


def plot_real_vs_pred(panels, path):
    """`panels`: lista de dict(y, p, title, band, metrics). Hexbin + diagonal ideal + promedio predicho por tramo."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 5.6), sharey=True)
    axes = np.atleast_1d(axes)
    hb = None
    for ax, pn in zip(axes, panels):
        hb = _hexpanel(ax, pn["y"], pn["p"], pn["title"], pn.get("band", False), pn.get("metrics"))
    axes[0].set_ylabel("Popularidad predicha")
    cb = fig.colorbar(hb, ax=list(axes), pad=0.015, shrink=0.85); cb.set_label("canciones por celda (escala log)")
    fig.legend(handles=[Line2D([], [], color=INK2, linestyle="--", label="predicción perfecta (real = predicho)"),
                        Line2D([], [], color=INK, marker="o", markersize=4, label="promedio de lo predicho por tramo de popularidad real")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.45, -0.07))
    p = _save(fig, path); plt.close(fig); return p


def plot_diagnostico_linea(y, p_ini, p_fin, share_ini, value_ini, share_fin, path):
    """Modelo inicial (profundidad 15) vs. corregido: real vs. predicho y distribucion de las predicciones."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    _hexpanel(axes[0], y, p_ini, "Modelo inicial (max_depth = 15)")
    axes[0].set_ylabel("Popularidad predicha")
    axes[0].annotate(f"{_d(share_ini * 100, 0)}% de las canciones\nrecibe ≈ {_d(value_ini)}", xy=(70, value_ini), xytext=(48, 82), fontsize=10.5, color=INK,
                     arrowprops=dict(arrowstyle="->", color=INK))
    _hexpanel(axes[1], y, p_fin, "Modelo corregido (sin límite de profundidad)")
    ax = axes[2]
    bins = np.arange(0, 101, 1)
    ax.hist(p_ini, bins=bins, color=C_INICIAL, alpha=0.35, label="Inicial", histtype="stepfilled")
    ax.hist(p_ini, bins=bins, color=C_INICIAL, histtype="step", linewidth=1.6)
    ax.hist(p_fin, bins=bins, color=C_RESTO, alpha=0.35, label="Corregido", histtype="stepfilled")
    ax.hist(p_fin, bins=bins, color=C_RESTO, histtype="step", linewidth=1.6)
    ax.set_title("Distribución de las predicciones")
    ax.set_xlabel("Popularidad predicha"); ax.set_ylabel("Canciones")
    ax.yaxis.set_major_formatter(lambda v, _: _n(v))
    ax.legend(handles=[Patch(color=C_INICIAL, alpha=0.6, label=f"Inicial: pico de {_d(share_ini * 100, 0)}% en un valor"),
                       Patch(color=C_RESTO, alpha=0.6, label=f"Corregido: máximo {_d(share_fin * 100, 1)}% en un valor")], loc="upper right")
    axes[0].set_xlabel("Popularidad real"); axes[1].set_xlabel("Popularidad real")
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


def plot_ganancia(curves, path):
    """`curves`: lista de (nombre, color, x, y). Curva de ganancia vs. azar."""
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.plot([0, 100], [0, 100], color=INK2, linestyle="--", linewidth=1.4, label="Sin modelo (al azar)")
    for name, color, x, y in curves:
        ax.plot(x * 100, y * 100, color=color, linewidth=2.2, label=name)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("% de canciones priorizadas (las de mayor popularidad predicha)")
    ax.set_ylabel("% de los éxitos reales capturados")
    ax.set_title("Si Spotify priorizara canciones con el modelo")
    ax.legend(loc="lower right")
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


# --------------------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------------------
def plot_codo_silhouette(k_values, inertias, silhouettes, k_elegido, path):
    k_values = list(k_values)
    best = k_values[int(np.argmax(silhouettes))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.plot(k_values, inertias, color=C_RESTO, marker="o", linewidth=2)
    ax.set_title("Método del codo"); ax.set_xlabel("Número de clusters (K)"); ax.set_ylabel("Inercia")
    ax.set_xticks(k_values); ax.yaxis.set_major_formatter(lambda v, _: _n(v))
    ax = axes[1]
    ax.plot(k_values, silhouettes, color=C_RESTO, marker="o", linewidth=2)
    for k, s in zip(k_values, silhouettes):
        ax.text(k, s + 0.006, _d(s, 3), ha="center", fontsize=9, color=INK)
    ax.scatter([k_elegido], [silhouettes[k_values.index(k_elegido)]], s=140, facecolors="none", edgecolors=INK, linewidths=2, zorder=5)
    ax.set_title(f"Silhouette: máximo en K={best}, se elige K={k_elegido}"); ax.set_xlabel("Número de clusters (K)"); ax.set_ylabel("Silhouette score")
    ax.set_xticks(k_values); ax.margins(y=0.15)
    ax.yaxis.set_major_formatter(lambda v, _: _d(v, 2))
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


def plot_clusters_pca(X_pca, labels, is_zero, path, sample=15000, seed=42):
    """Un panel por cluster (azul sobre fondo gris) + un panel con las canciones de popularidad 0 (naranjo)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_pca), size=min(sample, len(X_pca)), replace=False)
    X, lab, z = X_pca[idx], np.asarray(labels)[idx], np.asarray(is_zero)[idx]
    ks = sorted(set(np.asarray(labels).tolist()))
    n_panels = len(ks) + 1
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.3 * nrows), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, k in zip(axes, ks):
        ax.scatter(X[:, 0], X[:, 1], s=4, color="#d8d7d2", linewidths=0)
        m = lab == k
        ax.scatter(X[m, 0], X[m, 1], s=5, color=C_RESTO, linewidths=0, alpha=0.8)
        ax.set_title(f"Cluster {k}  ({_d(np.mean(np.asarray(labels) == k) * 100, 1)}%)")
    ax = axes[len(ks)]
    ax.scatter(X[:, 0], X[:, 1], s=4, color="#d8d7d2", linewidths=0)
    ax.scatter(X[z, 0], X[z, 1], s=6, color=C_CERO, linewidths=0, alpha=0.85)
    ax.set_title("Canciones con popularidad 0")
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    for ax in axes[:n_panels]:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.supxlabel("Componente principal 1", color=INK2); fig.supylabel("Componente principal 2", color=INK2)
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


def plot_clusters_ceros(profile, path):
    """`profile`: indice = cluster; columnas pop_con_ceros, pop_sin_ceros, pct_ceros."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    x = np.arange(len(profile))
    ax.bar(x - 0.2, profile["pop_con_ceros"], 0.38, color=C_CERO, label="Con ceros")
    ax.bar(x + 0.2, profile["pop_sin_ceros"], 0.38, color=C_RESTO, label="Sin ceros")
    for xi, a, b in zip(x, profile["pop_con_ceros"], profile["pop_sin_ceros"]):
        ax.text(xi - 0.2, a, _d(a), ha="center", va="bottom", fontsize=9.5)
        ax.text(xi + 0.2, b, _d(b), ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels([f"Cluster {c}" for c in profile.index])
    ax.set_title("Popularidad media por cluster"); ax.legend(loc="upper right", ncol=2); ax.grid(axis="x", visible=False); ax.set_ylim(0, profile[["pop_con_ceros", "pop_sin_ceros"]].to_numpy().max() * 1.28)
    ax = axes[1]
    ax.bar(x, profile["pct_ceros"], 0.6, color=C_CERO)
    for xi, v in zip(x, profile["pct_ceros"]):
        ax.text(xi, v, f"{_d(v)}%", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{c}" for c in profile.index]); ax.set_xlabel("Cluster")
    ax.set_title("% de canciones con popularidad 0"); ax.grid(axis="x", visible=False); ax.margins(y=0.15)
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


# --------------------------------------------------------------------------------------
# Diagramas de cajas (boxplots): mediana, 50% central de las canciones y rango tipico
# --------------------------------------------------------------------------------------
def plot_boxplot_audio_ceros(df, cols, path):
    """Diagramas de cajas de variables de audio: canciones con popularidad 0 vs. el resto."""
    zero = df["es_cero"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4))
    for ax, col in zip(axes.ravel(), cols):
        a = df.loc[zero, col].astype(float).dropna()
        b = df.loc[~zero, col].astype(float).dropna()
        bp = ax.boxplot([a, b], vert=False, widths=0.55, patch_artist=True, showfliers=False, whis=(5, 95))
        for i, (color, box) in enumerate(zip((C_CERO, C_RESTO), bp["boxes"])):
            box.set(facecolor=color, edgecolor=color, alpha=0.55, linewidth=1.6)
            for line in (bp["whiskers"][2 * i], bp["whiskers"][2 * i + 1], bp["caps"][2 * i], bp["caps"][2 * i + 1]):
                line.set(color=color, linewidth=1.4)
        for med in bp["medians"]:
            med.set(color=INK, linewidth=2)
        ax.set_yticks([1, 2]); ax.set_yticklabels(["Pop. = 0", "Pop. > 0"])
        ax.set_title(col)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"mediana: ceros {_d(a.median(), 2)}  ·  resto {_d(b.median(), 2)}", fontsize=10, color=INK2)
    fig.suptitle("Las cajas de los ceros y del resto se solapan: la diferencia es pequeña", x=0.01, y=0.995, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.925, "Caja = 50% central de las canciones · línea negra = mediana · bigotes = del 5% al 95%", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.90)); p = _save(fig, path); plt.close(fig); return p


def _paired_boxes(ax, groups, labels, pos_labels=None, horizontal=True):
    """Dibuja, para cada grupo, una caja CON ceros (naranjo) y otra SIN ceros (azul)."""
    n = len(groups)
    for k, (con, sin) in enumerate(groups):
        for off, data, color in ((-0.2, con, C_CERO), (0.2, sin, C_RESTO)):
            bp = ax.boxplot([data], positions=[k + off], vert=not horizontal, widths=0.32, patch_artist=True,
                            showfliers=False, whis=(5, 95), manage_ticks=False)
            bp["boxes"][0].set(facecolor=color, edgecolor=color, alpha=0.55, linewidth=1.5)
            for line in bp["whiskers"] + bp["caps"]:
                line.set(color=color, linewidth=1.3)
            bp["medians"][0].set(color=INK, linewidth=2)
    ticks = np.arange(n)
    if horizontal:
        ax.set_yticks(ticks); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    else:
        ax.set_xticks(ticks); ax.set_xticklabels(labels); ax.grid(axis="x", visible=False)


def plot_boxplot_genero(df, genres, path):
    """Popularidad por genero (los que tienen mas ceros): caja CON ceros vs. caja SIN ceros."""
    groups, labels = [], []
    for g in genres:
        d = df[df["track_genre"] == g]["popularity"]
        groups.append((d, d[d > 0]))
        labels.append(f"{g}  ({_d((d == 0).mean() * 100, 0)}% ceros)")
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(genres) + 1.8))
    _paired_boxes(ax, groups, labels, horizontal=True)
    ax.set_xlabel("Popularidad"); ax.set_xlim(-2, 100)
    ax.set_title("Géneros con más ceros: al quitarlos la caja se desplaza hacia arriba")
    ax.legend(handles=[Patch(facecolor=C_CERO, alpha=0.55, edgecolor=C_CERO, label="Con ceros"),
                       Patch(facecolor=C_RESTO, alpha=0.55, edgecolor=C_RESTO, label="Sin ceros")], loc="lower right")
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


def plot_boxplot_cluster(df, path):
    """Popularidad por cluster (columnas `cluster`, `popularity`, `es_cero`): caja CON ceros vs. SIN ceros."""
    clusters = sorted(df["cluster"].unique())
    groups = [(df.loc[df["cluster"] == c, "popularity"], df.loc[(df["cluster"] == c) & ~df["es_cero"], "popularity"]) for c in clusters]
    fig, ax = plt.subplots(figsize=(10, 5))
    _paired_boxes(ax, groups, [f"Cluster {c}" for c in clusters], horizontal=False)
    ax.set_ylabel("Popularidad"); ax.set_ylim(-2, 100)
    ax.set_title("Popularidad por cluster: con ceros las cajas bajan, sobre todo en el cluster 4")
    ax.legend(handles=[Patch(facecolor=C_CERO, alpha=0.55, edgecolor=C_CERO, label="Con ceros"),
                       Patch(facecolor=C_RESTO, alpha=0.55, edgecolor=C_RESTO, label="Sin ceros")], loc="upper right", ncol=2)
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


# --------------------------------------------------------------------------------------
# Variante multi-hot (un genero por cancion vs. todos sus generos)
# --------------------------------------------------------------------------------------
C_UNO = "#b9b7b1"
C_MULTI = "#52514e"


def plot_multihot(comp, path):
    """`comp`: columnas Modelo, Escenario, Version ('Un género' / 'Multi-hot'), R2, MAE."""
    models = list(dict.fromkeys(comp["Modelo"]))
    esc = [e for e, _, _ in SCENARIOS]
    fig, axes = plt.subplots(1, len(models), figsize=(13, 4.6), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        w = 0.36
        for j, (ver, color) in enumerate((("Un género", C_UNO), ("Multi-hot", C_MULTI))):
            vals = [comp[(comp["Modelo"] == m) & (comp["Escenario"] == e) & (comp["Version"] == ver)]["R2"].iloc[0] for e in esc]
            xs = np.arange(len(esc)) + (j - 0.5) * (w + 0.02)
            ax.bar(xs, vals, w, color=color, edgecolor="white", linewidth=1.2, label=ver)
            for x, v in zip(xs, vals):
                ax.text(x, v, _d(v, 2), ha="center", va="bottom", fontsize=9.5)
        ax.set_xticks(np.arange(len(esc)))
        ax.set_xticklabels([e.replace(" · ", "\n") for e in esc], fontsize=9.5)
        ax.set_title(m.replace("Regresion", "Regresión")); ax.grid(axis="x", visible=False); ax.margins(y=0.15)
        ax.yaxis.set_major_formatter(lambda v, _: _d(v, 1))
    np.atleast_1d(axes)[0].set_ylabel("R² (mayor es mejor)")
    np.atleast_1d(axes)[-1].legend(loc="upper right")
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


# Compatibilidad con versiones anteriores del proyecto
def save_metrics_plot(results, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m in zip(axes, ["MAE", "RMSE", "R2"]):
        ax.bar(results["Modelo"], results[m], color=C_RESTO); ax.set_title(m)
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p


def save_scatter(y_true, y_pred, title, xlabel, ylabel, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, color=C_RESTO)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.tight_layout(); p = _save(fig, path); plt.close(fig); return p
