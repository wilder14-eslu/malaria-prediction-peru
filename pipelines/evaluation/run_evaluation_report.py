"""Reporte de evaluacion estadistica del modelo campeon (LightGBM por cuantiles)
frente a los baselines, en validacion (2019-2020) y en el hold-out de test
(2021-2024, NUNCA usado para seleccionar modelo).

Genera:
- ``docs/images/results/*.png``: graficas del reporte (EDA + evaluacion).
- ``docs/results/metrics.json``: todas las metricas en formato maquina.
- ``docs/results/*.csv``: tablas de metricas (por horizonte, estrato, departamento).

Metricas:
- Puntuales (sobre la mediana q0.5): MAE, RMSE, WAPE, sesgo %, skill vs naive.
- Probabilisticas: pinball loss por cuantil, WIS (Weighted Interval Score,
  metrica oficial de los hubs de forecasting del CDC), cobertura empirica del
  intervalo 80 %, ancho medio del intervalo (nitidez / sharpness) y
  calibracion por cuantil.
- Inferencia: test de Diebold-Mariano (HAC Newey-West + correccion de
  Harvey-Leybourne-Newbold) y bootstrap por bloques de semanas para el IC 95 %
  de la mejora en WAPE.
- Decision: Precision@20 (de los 20 distritos con mas casos predichos por
  semana, cuantos estan realmente en el top 20).

Uso: ``python -m pipelines.evaluation.run_evaluation_report``
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

from src.features.pipeline import build_features  # noqa: E402
from src.features.spatial import add_neighbor_lag_features, load_adjacency  # noqa: E402
from src.features.targets import create_forecast_targets  # noqa: E402
from src.models.baseline import BASELINE_PREDICTORS  # noqa: E402
from src.models.forecasting import get_feature_columns, predict_quantiles  # noqa: E402
from src.preprocessing.temporal_split import walk_forward_split  # noqa: E402

IMG_DIR = Path("docs/images/results")
RES_DIR = Path("docs/results")
MODELS_DIR = Path("models")
SEED = 42
N_BOOT = 1000
BLOCK = 8  # semanas por bloque en el bootstrap (preserva autocorrelacion)
TOP_K = 20

# Paleta (validada, CVD-safe): azul = campeon, naranja/aqua/amarillo = baselines
C_MODEL = "#2a78d6"
C_BASE = {"naive": "#eb6834", "moving_average_4": "#1baf7a", "seasonal_naive": "#eda100"}
C_GRAY = "#8a8984"
C_TEXT = "#0b0b0b"
C_TEXT2 = "#52514e"
C_GRID = "#e6e5e0"
C_SURF = "#fcfcfb"
LABELS = {
    "lightgbm_quantile": "LightGBM cuantiles (campeón)",
    "naive": "Naive",
    "moving_average_4": "Media móvil 4s",
    "seasonal_naive": "Naive estacional",
}
COLORS = {"lightgbm_quantile": C_MODEL, **C_BASE}

plt.rcParams.update(
    {
        "figure.facecolor": C_SURF,
        "axes.facecolor": C_SURF,
        "savefig.facecolor": C_SURF,
        "axes.edgecolor": C_GRID,
        "axes.labelcolor": C_TEXT2,
        "axes.titlecolor": C_TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.grid": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": C_TEXT2,
        "ytick.color": C_TEXT2,
        "font.size": 10,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "figure.dpi": 110,
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(IMG_DIR / name, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def wape(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).sum() / np.abs(y).sum())


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def wis(
    y: np.ndarray, lo: np.ndarray, med: np.ndarray, hi: np.ndarray, alpha: float = 0.2
) -> float:
    """Weighted Interval Score (Bracher et al., 2021) con 1 intervalo + mediana."""
    interval = (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    score = (0.5 * np.abs(y - med) + (alpha / 2) * interval) / 1.5
    return float(np.mean(score))


def diebold_mariano(d: np.ndarray, h: int) -> tuple[float, float]:
    """DM con varianza HAC (Newey-West, kernel Bartlett) y correccion HLN.
    ``d`` = diferencial de perdida semanal (modelo - baseline); negativo = modelo mejor.
    """
    t = len(d)
    lag = max(h - 1, int(np.floor(4 * (t / 100) ** (2 / 9))))
    dc = d - d.mean()
    gamma0 = np.dot(dc, dc) / t
    var = gamma0
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        var += 2 * w * np.dot(dc[k:], dc[:-k]) / t
    dm = d.mean() / np.sqrt(var / t)
    hln = np.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
    stat = dm * hln
    p = 2 * stats.t.sf(np.abs(stat), df=t - 1)
    return float(stat), float(p)


def block_bootstrap_wape(
    week_ids: np.ndarray, y: np.ndarray, p_model: np.ndarray, p_base: np.ndarray, rng
) -> dict[str, float]:
    """IC 95 % de WAPE del modelo y de la mejora relativa, remuestreando bloques
    de semanas completas (todos los distritos de una semana van juntos)."""
    weeks = np.unique(week_ids)
    idx_by_week = {w: np.where(week_ids == w)[0] for w in weeks}
    n_blocks = int(np.ceil(len(weeks) / BLOCK))
    starts = np.arange(len(weeks) - BLOCK + 1)
    w_m, rel = [], []
    for _ in range(N_BOOT):
        s = rng.choice(starts, n_blocks)
        sel_weeks = np.concatenate([weeks[i : i + BLOCK] for i in s])
        idx = np.concatenate([idx_by_week[w] for w in sel_weeks])
        wm = wape(y[idx], p_model[idx])
        wb = wape(y[idx], p_base[idx])
        w_m.append(wm)
        rel.append(1 - wm / wb)
    return {
        "wape_ci_low": float(np.percentile(w_m, 2.5)),
        "wape_ci_high": float(np.percentile(w_m, 97.5)),
        "improvement_ci_low": float(np.percentile(rel, 2.5)),
        "improvement_ci_high": float(np.percentile(rel, 97.5)),
    }


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict, pd.DataFrame]:
    with open("configs/data.yaml", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/model.yaml", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    canonical = pd.read_parquet(data_cfg["gold"]["canonical_weekly_cases"])
    feats = build_features(canonical)
    feats = add_neighbor_lag_features(
        feats, load_adjacency("data/reference/district_adjacency.csv")
    )
    horizons = tuple(model_cfg["forecasting"]["horizons"])
    feats = create_forecast_targets(feats, target_col="cases_total", horizons=horizons)
    feature_cols = get_feature_columns(feats, horizons=horizons)
    return canonical, feats, feature_cols, model_cfg, feats


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def eda(canonical: pd.DataFrame, feats: pd.DataFrame, cfg: dict) -> dict:
    v = cfg["validation"]
    out: dict = {}

    # 01 Serie nacional semanal con particion walk-forward
    nat = feats.groupby("epi_index").agg(
        y=("cases_total", "sum"), year=("epi_year", "first"), week=("epi_week", "first")
    )
    x = nat["year"] + (nat["week"] - 1) / 52
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axvspan(v["train_end_year"] + 1, v["val_end_year"] + 1, color="#eda100", alpha=0.12, lw=0)
    ax.axvspan(v["val_end_year"] + 1, v["test_end_year"] + 1, color="#1baf7a", alpha=0.12, lw=0)
    ax.plot(x, nat["y"], color=C_MODEL, lw=1.4)
    ax.plot(
        x, nat["y"].rolling(13, center=True).mean(), color=C_TEXT, lw=1.6, label="Media móvil 13 s"
    )
    ymax = nat["y"].max()
    for x0, lab in [
        (2000.3, "Train 2000–2018"),
        (2019.1, "Val\n2019–20"),
        (2021.1, "Test (hold-out)\n2021–24"),
    ]:
        ax.text(x0, ymax * 0.97, lab, va="top", fontsize=9, color=C_TEXT2)
    ax.set_title("Casos semanales de malaria en Perú (2000–2024) y partición walk-forward")
    ax.set_ylabel("Casos / semana")
    ax.legend(loc="upper center")
    ax.set_xlim(2000, 2025)
    _save(fig, "01_serie_nacional.png")

    # 02 Distribucion: inflacion de ceros y sobredispersion
    y = feats["cases_total"].to_numpy()
    zero_share = float((y == 0).mean())
    mean, var = float(y.mean()), float(y.var())
    pos = y[y > 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(
        ["0 casos", "1–5", "6–20", "21–100", ">100"],
        [
            np.mean(y == 0),
            np.mean((y >= 1) & (y <= 5)),
            np.mean((y > 5) & (y <= 20)),
            np.mean((y > 20) & (y <= 100)),
            np.mean(y > 100),
        ],
        color=C_MODEL,
        width=0.6,
    )
    axes[0].set_title("Distribución de casos por distrito-semana")
    axes[0].set_ylabel("% de observaciones")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    for val in axes[0].patches:
        axes[0].text(
            val.get_x() + val.get_width() / 2,
            val.get_height(),
            f"{val.get_height():.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=C_TEXT2,
        )
    axes[1].hist(np.log10(pos), bins=40, color=C_MODEL, edgecolor=C_SURF, linewidth=1)
    axes[1].set_title("Semanas con casos (> 0): cola pesada")
    axes[1].set_xlabel("log10(casos)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].text(
        0.98,
        0.95,
        f"media = {mean:.2f}\nvarianza = {var:,.0f}\nvar/media = {var / mean:,.0f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=C_TEXT2,
    )
    _save(fig, "02_distribucion_ceros.png")
    out["zero_share"] = zero_share
    out["mean"] = mean
    out["variance"] = var
    out["dispersion_index"] = var / mean

    # 03 Concentracion (curva de Lorenz / Pareto) por distrito
    tot = canonical.groupby("ubigeo")["cases_total"].sum().sort_values(ascending=False)
    cum = tot.cumsum() / tot.sum()
    share_d = np.arange(1, len(tot) + 1) / len(tot)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(share_d, cum.values, color=C_MODEL)
    ax.plot([0, 1], [0, 1], color=C_GRAY, lw=1, ls="--", label="Igualdad perfecta")
    top10 = float(cum.iloc[int(0.10 * len(tot)) - 1])
    top1 = float(cum.iloc[int(0.01 * len(tot)) - 1])
    ax.scatter([0.10], [top10], color=C_MODEL, s=40, zorder=3, edgecolor=C_SURF, linewidth=2)
    ax.annotate(
        f"10 % de distritos = {top10:.0%} de los casos",
        (0.10, top10),
        (0.2, 0.6),
        fontsize=9,
        color=C_TEXT2,
        arrowprops={"arrowstyle": "-", "color": C_GRAY},
    )
    lorenz = np.concatenate([[0], np.sort(tot.values).cumsum() / tot.sum()])
    gini = float(1 - 2 * np.trapezoid(lorenz, dx=1 / len(tot)))
    ax.set_title(f"Concentración espacial de casos (Gini = {gini:.2f})")
    ax.set_xlabel("% acumulado de distritos (de mayor a menor carga)")
    ax.set_ylabel("% acumulado de casos")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.legend(loc="lower right")
    _save(fig, "03_concentracion_pareto.png")
    out.update(
        {
            "gini_districts": gini,
            "top10pct_share": top10,
            "top1pct_share": top1,
            "n_districts": int(len(tot)),
        }
    )

    # 04 Departamentos
    dep = canonical.groupby("departamento")["cases_total"].sum().sort_values()
    dep_share = dep / dep.sum()
    top = dep_share.tail(10)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(top.index, top.values, color=C_MODEL, height=0.6)
    for i, val in enumerate(top.values):
        ax.text(val, i, f" {val:.1%}", va="center", fontsize=9, color=C_TEXT2)
    ax.set_title("Top 10 departamentos por % de casos (2000–2024)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis="y", visible=False)
    _save(fig, "04_departamentos.png")
    out["department_share"] = {
        k: float(v) for k, v in dep_share.sort_values(ascending=False).head(10).items()
    }

    # 05 Estacionalidad: perfil por semana epidemiologica (indice relativo)
    w = feats.groupby(["epi_year", "epi_week"])["cases_total"].sum().reset_index()
    w["rel"] = w["cases_total"] / w.groupby("epi_year")["cases_total"].transform("mean")
    prof = w.groupby("epi_week")["rel"].agg(
        ["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)]
    )
    prof.columns = ["med", "q25", "q75"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.fill_between(
        prof.index,
        prof["q25"],
        prof["q75"],
        color=C_MODEL,
        alpha=0.18,
        lw=0,
        label="Rango intercuartil entre años",
    )
    ax.plot(prof.index, prof["med"], color=C_MODEL, label="Mediana")
    ax.axhline(1, color=C_GRAY, lw=1, ls="--")
    ax.set_title("Estacionalidad: casos de la semana / promedio semanal del año")
    ax.set_xlabel("Semana epidemiológica")
    ax.set_ylabel("Índice estacional")
    ax.legend(loc="upper right")
    _save(fig, "05_estacionalidad.png")
    out["seasonal_peak_week"] = int(prof["med"].idxmax())
    out["seasonal_trough_week"] = int(prof["med"].idxmin())

    # 06 Composicion por especie
    sp = canonical.groupby("epi_year")[["cases_vivax", "cases_falciparum"]].sum()
    sp_share = sp.div(sp.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(sp_share.index, sp_share["cases_vivax"], color=C_MODEL, width=0.8, label="P. vivax")
    ax.bar(
        sp_share.index,
        sp_share["cases_falciparum"],
        bottom=sp_share["cases_vivax"],
        color="#eb6834",
        width=0.8,
        label="P. falciparum",
        edgecolor=C_SURF,
        linewidth=1,
    )
    ax.set_title("Composición por especie de Plasmodium")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.legend(loc="lower left", ncol=2)
    ax.set_ylim(0, 1)
    _save(fig, "06_especies.png")
    out["vivax_share_total"] = float(sp["cases_vivax"].sum() / sp.sum().sum())

    # Autocorrelacion de la serie nacional (justifica lags)
    s = nat["y"].to_numpy(dtype=float)
    s = s - s.mean()
    acf = [float(np.dot(s[k:], s[:-k]) / np.dot(s, s)) for k in (1, 2, 4, 52)]
    out["acf_national"] = dict(zip(["lag1", "lag2", "lag4", "lag52"], acf, strict=True))
    return out


# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------
def predictions_for(split: pd.DataFrame, feature_cols: list[str], h: int) -> pd.DataFrame:
    target = f"target_h{h}"
    models = joblib.load(MODELS_DIR / f"forecasting_h{h}_champion.joblib")
    q = predict_quantiles(models, split, feature_cols)
    df = split[
        ["ubigeo", "departamento", "distrito", "epi_index", "epi_year", "epi_week", target]
    ].copy()
    df = df.rename(columns={target: "y"})
    df[["q0.1", "q0.5", "q0.9"]] = q[["q0.1", "q0.5", "q0.9"]].clip(lower=0)
    for name, fn in BASELINE_PREDICTORS.items():
        df[name] = fn(split, horizons=(h,))[h]
    # Muestra comun para la comparacion principal: target + campeon + naive +
    # media movil. El naive estacional necesita 52 semanas de historia por
    # distrito; exigirlo recortaria la muestra, asi que se evalua en su propio
    # subconjunto (ver ``evaluate``).
    return df.dropna(subset=["y", "q0.5", "naive", "moving_average_4"])


def evaluate(feats: pd.DataFrame, feature_cols: list[str], cfg: dict) -> tuple[dict, dict]:
    v = cfg["validation"]
    _, val, test = walk_forward_split(
        feats, v["train_end_year"], v["val_end_year"], v["test_end_year"]
    )
    rng = np.random.default_rng(SEED)
    results: dict = {}
    preds_store: dict = {}
    for split_name, split in [("validation", val), ("test", test)]:
        results[split_name] = {}
        for h in cfg["forecasting"]["horizons"]:
            df = predictions_for(split, feature_cols, h)
            preds_store[(split_name, h)] = df
            y = df["y"].to_numpy(dtype=float)
            r: dict = {
                "n_obs": int(len(df)),
                "n_weeks": int(df["epi_index"].nunique()),
                "n_districts": int(df["ubigeo"].nunique()),
                "models": {},
            }
            for name in ["lightgbm_quantile", *BASELINE_PREDICTORS]:
                col = "q0.5" if name == "lightgbm_quantile" else name
                mask = df[col].notna().to_numpy()
                yy = y[mask]
                p = df.loc[mask, col].to_numpy(dtype=float)
                m = {
                    "mae": float(np.mean(np.abs(yy - p))),
                    "rmse": float(np.sqrt(np.mean((yy - p) ** 2))),
                    "wape": wape(yy, p),
                    "bias_pct": float((p - yy).sum() / yy.sum()),
                    "n_obs": int(mask.sum()),
                }
                naive_ref = float(np.mean(np.abs(yy - df.loc[mask, "naive"].to_numpy(float))))
                m["skill_vs_naive"] = 1 - m["mae"] / naive_ref
                r["models"][name] = m
            lo, med, hi = (df[c].to_numpy(dtype=float) for c in ("q0.1", "q0.5", "q0.9"))
            inside = (y >= lo) & (y <= hi)
            posm = y > 0
            r["probabilistic"] = {
                "pinball_q0.1": pinball(y, lo, 0.1),
                "pinball_q0.5": pinball(y, med, 0.5),
                "pinball_q0.9": pinball(y, hi, 0.9),
                "wis": wis(y, lo, med, hi),
                "coverage_80": float(inside.mean()),
                "coverage_80_y_pos": float(inside[posm].mean()),
                "coverage_80_y_zero": float(inside[~posm].mean()),
                "share_y_zero": float((~posm).mean()),
                "mean_width_80": float(np.mean(hi - lo)),
                "mean_width_80_y_pos": float(np.mean((hi - lo)[posm])),
                "hit_below_q0.1": float(np.mean(y < lo)),
                "hit_below_q0.5": float(np.mean(y < med)),
                "hit_below_q0.9": float(np.mean(y < hi)),
                "hit_le_q0.1": float(np.mean(y <= lo)),
                "hit_le_q0.5": float(np.mean(y <= med)),
                "hit_le_q0.9": float(np.mean(y <= hi)),
            }
            # Mejor baseline por WAPE en este split/horizonte
            best = min(BASELINE_PREDICTORS, key=lambda n: r["models"][n]["wape"])
            r["best_baseline"] = best
            wk = df.groupby("epi_index").apply(
                lambda g, b=best: pd.Series(
                    {"m": np.abs(g["y"] - g["q0.5"]).sum(), "b": np.abs(g["y"] - g[b]).sum()}
                ),
                include_groups=False,
            )
            stat, pval = diebold_mariano((wk["m"] - wk["b"]).to_numpy(), h)
            r["diebold_mariano"] = {
                "vs": best,
                "loss": "error absoluto agregado semanal",
                "stat": stat,
                "p_value": pval,
            }
            r["bootstrap"] = block_bootstrap_wape(
                df["epi_index"].to_numpy(), y, med, df[best].to_numpy(dtype=float), rng
            )
            r["improvement_vs_best_baseline"] = (
                1 - r["models"]["lightgbm_quantile"]["wape"] / r["models"][best]["wape"]
            )
            # Precision@K semanal (decision: a que distritos mandar recursos)
            precs = {"lightgbm_quantile": [], best: []}
            for _, g in df.groupby("epi_index"):
                if (g["y"] > 0).sum() < TOP_K:
                    continue
                true_top = set(g.nlargest(TOP_K, "y")["ubigeo"])
                precs["lightgbm_quantile"].append(
                    len(true_top & set(g.nlargest(TOP_K, "q0.5")["ubigeo"])) / TOP_K
                )
                precs[best].append(len(true_top & set(g.nlargest(TOP_K, best)["ubigeo"])) / TOP_K)
            r["precision_at_20"] = {k: float(np.mean(vv)) for k, vv in precs.items()}
            results[split_name][f"h{h}"] = r
            print(
                f"{split_name} h{h}: WAPE={r['models']['lightgbm_quantile']['wape']:.3f} "
                f"best={best} {r['models'][best]['wape']:.3f} DM p={pval:.2g}"
            )
    return results, preds_store


# ---------------------------------------------------------------------------
# Graficas de evaluacion
# ---------------------------------------------------------------------------
def plot_eval(results: dict, preds: dict, feature_cols: list[str]) -> dict:
    extra: dict = {}
    hs = [1, 2, 3, 4]
    order = ["lightgbm_quantile", "naive", "moving_average_4", "seasonal_naive"]

    # 07 WAPE por horizonte (val y test)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, split in zip(axes, ["validation", "test"], strict=True):
        width = 0.2
        for i, name in enumerate(order):
            vals = [results[split][f"h{h}"]["models"][name]["wape"] for h in hs]
            xs = np.arange(len(hs)) + (i - 1.5) * width
            ax.bar(xs, vals, width=width * 0.9, color=COLORS[name], label=LABELS[name])
            if name == "lightgbm_quantile":
                for xx, vv in zip(xs, vals, strict=True):
                    ax.text(xx, vv, f"{vv:.2f}", ha="center", va="bottom", fontsize=8, color=C_TEXT)
        ax.set_xticks(np.arange(len(hs)), [f"h = {h} sem" for h in hs])
        ax.set_title("Validación 2019–2020" if split == "validation" else "Test hold-out 2021–2024")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("WAPE (menor es mejor)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    _save(fig, "07_wape_horizonte.png")

    # 08 Mejora vs mejor baseline con IC 95 % bootstrap
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for split, col, off in [("validation", "#eda100", -0.1), ("test", C_MODEL, 0.1)]:
        imp = [results[split][f"h{h}"]["improvement_vs_best_baseline"] for h in hs]
        lo = [results[split][f"h{h}"]["bootstrap"]["improvement_ci_low"] for h in hs]
        hi = [results[split][f"h{h}"]["bootstrap"]["improvement_ci_high"] for h in hs]
        xs = np.array(hs) + off
        ax.errorbar(
            xs,
            imp,
            yerr=[np.array(imp) - lo, np.array(hi) - imp],
            fmt="o",
            color=col,
            ms=8,
            capsize=4,
            lw=2,
            mec=C_SURF,
            mew=2,
            label="Validación" if split == "validation" else "Test hold-out",
        )
        for xx, vv in zip(xs, imp, strict=True):
            ax.text(xx + 0.06, vv, f"{vv:.0%}", va="center", fontsize=8.5, color=C_TEXT2)
    ax.axhline(0, color=C_GRAY, lw=1, ls="--")
    ax.set_xticks(hs, [f"h = {h}" for h in hs])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_title("Mejora en WAPE vs. mejor baseline (IC 95 % bootstrap por bloques)")
    ax.set_ylabel("1 − WAPE modelo / WAPE baseline")
    ax.legend(loc="lower right")
    _save(fig, "08_mejora_bootstrap.png")

    # 09 Calibracion por cuantil (test, h1). Con datos de conteo (discretos y
    # con masa en 0) P(y < q) y P(y <= q) difieren mucho: la calibracion
    # correcta de un cuantil tau cumple P(y < q) <= tau <= P(y <= q).
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    nominal = [0.1, 0.5, 0.9]
    p = results["test"]["h1"]["probabilistic"]
    lo_ = [p["hit_below_q0.1"], p["hit_below_q0.5"], p["hit_below_q0.9"]]
    hi_ = [p["hit_le_q0.1"], p["hit_le_q0.5"], p["hit_le_q0.9"]]
    ax.plot([0, 1], [0, 1], color=C_GRAY, lw=1, ls="--", label="Calibración perfecta (τ)")
    for t, a, b in zip(nominal, lo_, hi_, strict=True):
        ok = a <= t <= b
        ax.vlines(t, a, b, color=C_MODEL if ok else "#eb6834", lw=6, alpha=0.35)
        ax.text(
            t + 0.03,
            (a + b) / 2,
            f"[{a:.0%}, {b:.0%}]\n{'contiene τ' if ok else 'no contiene τ'}",
            fontsize=8.5,
            color=C_TEXT2,
            va="center",
        )
    ax.plot(nominal, lo_, marker="o", color=C_MODEL, ms=7, mec=C_SURF, mew=2, label="P(y < q̂τ)")
    ax.plot(
        nominal,
        hi_,
        marker="s",
        color=C_TEXT,
        ms=6,
        ls=":",
        lw=1.5,
        mec=C_SURF,
        mew=1.5,
        label="P(y ≤ q̂τ)",
    )
    ax.set_xlabel("Cuantil nominal τ")
    ax.set_ylabel("Frecuencia empírica")
    ax.set_title("Calibración de cuantiles para conteos (test, h = 1)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 1.02)
    _save(fig, "09_calibracion_cuantiles.png")

    # 10 Cobertura del intervalo 80 %: total vs y > 0 vs y = 0
    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.25
    series = [
        ("coverage_80", "Todas las semanas", C_MODEL),
        ("coverage_80_y_pos", "Semanas con casos (y > 0)", "#eb6834"),
        ("coverage_80_y_zero", "Semanas sin casos (y = 0)", C_GRAY),
    ]
    for i, (k, lab, col) in enumerate(series):
        vals = [results["test"][f"h{h}"]["probabilistic"][k] for h in hs]
        xs = np.arange(4) + (i - 1) * width
        ax.bar(xs, vals, width=width * 0.9, color=col, label=lab)
        for xx, vv in zip(xs, vals, strict=True):
            ax.text(xx, vv, f"{vv:.0%}", ha="center", va="bottom", fontsize=8, color=C_TEXT2)
    ax.axhline(0.8, color=C_TEXT, lw=1.2, ls="--")
    ax.text(-0.45, 0.815, "nominal 80 %", fontsize=8.5, color=C_TEXT, ha="left", va="bottom")
    ax.set_xticks(range(4), [f"h = {h}" for h in hs])
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_title("Cobertura empírica del intervalo q0.1–q0.9 (test)")
    ax.legend(loc="upper center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    _save(fig, "10_cobertura_intervalo.png")

    # 11 Error por estrato de actividad (test, h1)
    df = preds[("test", 1)].copy()
    bins = [-0.5, 0.5, 5.5, 20.5, 100.5, np.inf]
    labs = ["0", "1–5", "6–20", "21–100", ">100"]
    df["estrato"] = pd.cut(df["y"], bins=bins, labels=labs)
    rows = []
    for e, g in df.groupby("estrato", observed=True):
        rows.append(
            {
                "estrato": e,
                "n": len(g),
                "share_cases": g["y"].sum() / df["y"].sum(),
                "mae_model": np.mean(np.abs(g["y"] - g["q0.5"])),
                "mae_naive": np.mean(np.abs(g["y"] - g["naive"])),
                "bias_model": np.mean(g["q0.5"] - g["y"]),
                "coverage_80": np.mean((g["y"] >= g["q0.1"]) & (g["y"] <= g["q0.9"])),
            }
        )
    strata = pd.DataFrame(rows)
    strata.to_csv(RES_DIR / "error_por_estrato_test_h1.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    xs = np.arange(len(strata))
    ax.bar(xs - 0.2, strata["mae_model"], 0.38, color=C_MODEL, label=LABELS["lightgbm_quantile"])
    ax.bar(xs + 0.2, strata["mae_naive"], 0.38, color=C_BASE["naive"], label="Naive")
    ax.set_yscale("log")
    ax.set_xticks(
        xs, [f"{e}\n(n={n:,})" for e, n in zip(strata["estrato"], strata["n"], strict=True)]
    )
    ax.set_xlabel("Casos reales en la semana objetivo")
    ax.set_ylabel("MAE (escala log)")
    ax.set_title("Error absoluto medio por estrato de actividad (test, h = 1)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    _save(fig, "11_error_por_estrato.png")
    extra["strata_test_h1"] = strata.assign(estrato=strata["estrato"].astype(str)).to_dict(
        orient="records"
    )

    # 12 WAPE por departamento (test, h1) top 8 por volumen
    dep_rows = []
    for dname, g in df.groupby("departamento"):
        if g["y"].sum() == 0:
            continue
        dep_rows.append(
            {
                "departamento": dname,
                "casos": int(g["y"].sum()),
                "wape_model": wape(g["y"].to_numpy(float), g["q0.5"].to_numpy(float)),
                "wape_naive": wape(g["y"].to_numpy(float), g["naive"].to_numpy(float)),
            }
        )
    deps = pd.DataFrame(dep_rows).sort_values("casos", ascending=False)
    deps.to_csv(RES_DIR / "wape_por_departamento_test_h1.csv", index=False)
    top = deps.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ys = np.arange(len(top))
    ax.barh(ys + 0.2, top["wape_model"], 0.38, color=C_MODEL, label=LABELS["lightgbm_quantile"])
    ax.barh(ys - 0.2, top["wape_naive"], 0.38, color=C_BASE["naive"], label="Naive")
    ax.set_yticks(
        ys,
        [
            f"{d.title()} ({c:,} casos)"
            for d, c in zip(top["departamento"], top["casos"], strict=True)
        ],
    )
    ax.set_xlabel("WAPE (menor es mejor)")
    ax.set_title("WAPE por departamento: top 8 por volumen (test, h = 1)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", visible=False)
    _save(fig, "12_wape_departamento.png")
    extra["departments_test_h1"] = deps.head(8).to_dict(orient="records")

    # 13 Fan chart del distrito con mas casos en test (h1)
    top_ub = df.groupby("ubigeo")["y"].sum().idxmax()
    g = df[df["ubigeo"] == top_ub].sort_values("epi_index")
    xs = g["epi_year"] + (g["epi_week"] - 1) / 52 + 1 / 52  # semana objetivo = t + 1
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(
        xs, g["q0.1"], g["q0.9"], color=C_MODEL, alpha=0.2, lw=0, label="Intervalo 80 % (q0.1–q0.9)"
    )
    ax.plot(xs, g["q0.5"], color=C_MODEL, label="Predicción mediana (q0.5)")
    ax.plot(xs, g["y"], color=C_TEXT, lw=1.3, label="Casos reales")
    name = f"{g['distrito'].iloc[0].title()} ({g['departamento'].iloc[0].title()}, UBIGEO {top_ub})"
    ax.set_title(f"Pronóstico a 1 semana vs. realidad: {name}, test 2021–2024")
    ax.set_ylabel("Casos / semana")
    ax.legend(loc="upper left", ncol=3)
    _save(fig, "13_fan_chart_distrito.png")
    gy = g["y"].to_numpy(float)
    extra["fan_district"] = {
        "ubigeo": top_ub,
        "name": name,
        "wape": wape(gy, g["q0.5"].to_numpy(float)),
        "coverage_80": float(np.mean((gy >= g["q0.1"]) & (gy <= g["q0.9"]))),
    }

    # 14 Serie nacional agregada: suma de medianas vs real (test, h1)
    nat = df.groupby("epi_index").agg(
        y=("y", "sum"),
        p=("q0.5", "sum"),
        n=("naive", "sum"),
        yr=("epi_year", "first"),
        wk=("epi_week", "first"),
    )
    xs = nat["yr"] + (nat["wk"] - 1) / 52 + 1 / 52
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(xs, nat["y"], color=C_TEXT, lw=1.3, label="Casos reales (nacional)")
    ax.plot(xs, nat["p"], color=C_MODEL, label="Σ medianas predichas")
    ax.set_title("Agregado nacional a 1 semana (test 2021–2024)")
    ax.set_ylabel("Casos / semana")
    ax.legend(loc="upper left")
    _save(fig, "14_agregado_nacional.png")
    r_nat = float(np.corrcoef(nat["y"], nat["p"])[0, 1])
    extra["national_test_h1"] = {
        "pearson_r": r_nat,
        "wape_national": float(np.abs(nat["y"] - nat["p"]).sum() / nat["y"].sum()),
        "bias_national": float((nat["p"] - nat["y"]).sum() / nat["y"].sum()),
    }

    # 15 Importancia de variables (gain, q0.5, h1)
    m = joblib.load(MODELS_DIR / "forecasting_h1_champion.joblib")[0.5]
    gain = pd.Series(
        m.booster_.feature_importance("gain"), index=m.booster_.feature_name()
    ).sort_values()
    gain = gain / gain.sum()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(gain.index, gain.values, color=C_MODEL, height=0.6)
    for i, val in enumerate(gain.values):
        ax.text(val, i, f" {val:.1%}", va="center", fontsize=8.5, color=C_TEXT2)
    ax.set_title("Importancia de variables (ganancia, modelo q0.5, h = 1)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis="y", visible=False)
    _save(fig, "15_importancia_variables.png")
    extra["feature_importance_gain_h1"] = {
        k: float(v) for k, v in gain.sort_values(ascending=False).items()
    }
    return extra


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    canonical, feats, feature_cols, cfg, _ = load_dataset()
    dup = int(feats.duplicated(["ubigeo", "epi_index"]).sum())
    eda_stats = eda(canonical, feats, cfg)
    eda_stats["rows_canonical"] = int(len(canonical))
    eda_stats["rows_weekly_grid"] = int(len(feats))
    eda_stats["duplicated_ubigeo_epi_index"] = dup
    eda_stats["rows_epi_week_53"] = int((canonical["epi_week"] == 53).sum())
    eda_stats["total_cases"] = int(canonical["cases_total"].sum())
    results, preds = evaluate(feats, feature_cols, cfg)
    extra = plot_eval(results, preds, feature_cols)

    rows = []
    for split, hs in results.items():
        for hk, r in hs.items():
            for name, m in r["models"].items():
                rows.append({"split": split, "horizon": hk, "model": name, **m})
    pd.DataFrame(rows).to_csv(RES_DIR / "metricas_puntuales.csv", index=False)
    prob = [
        {
            "split": s,
            "horizon": hk,
            **r["probabilistic"],
            "dm_stat": r["diebold_mariano"]["stat"],
            "dm_p": r["diebold_mariano"]["p_value"],
            **r["bootstrap"],
        }
        for s, hs in results.items()
        for hk, r in hs.items()
    ]
    pd.DataFrame(prob).to_csv(RES_DIR / "metricas_probabilisticas.csv", index=False)

    with open(RES_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "eda": eda_stats,
                "evaluation": results,
                "diagnostics": extra,
                "config": {"seed": SEED, "n_boot": N_BOOT, "block_weeks": BLOCK, "top_k": TOP_K},
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=float,
        )
    print(f"Reporte escrito en {RES_DIR} y {IMG_DIR}")


if __name__ == "__main__":
    main()
