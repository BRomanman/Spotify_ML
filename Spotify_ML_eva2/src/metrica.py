import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def metrics_dataframe(rows):
    return pd.DataFrame(rows)


def pct_within(y_true, y_pred, tolerance=10):
    """% de canciones cuya prediccion se equivoca por `tolerance` puntos de popularidad o menos."""
    return float((np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= tolerance).mean() * 100)


def gain_curve(y_true, y_pred, hit_frac=0.20):
    """Curva de ganancia: si se prioriza el X% de canciones con mayor prediccion, ¿que fraccion de los
    'exitos' reales (el `hit_frac` superior en popularidad) se captura? Devuelve (x, y)."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_pred), kind="stable")
    is_hit = (y_true >= np.quantile(y_true, 1 - hit_frac))[order]
    x = np.arange(1, len(y_true) + 1) / len(y_true)
    y = np.cumsum(is_hit) / is_hit.sum()
    return np.insert(x, 0, 0.0), np.insert(y, 0, 0.0)


def business_kpis(y_true, y_pred, top_frac=0.20, hit_frac=0.20, tolerance=10):
    """KPIs entendibles para negocio.

    - pct_dentro_10: % de canciones con error <= 10 puntos.
    - precision_top: de las canciones que el modelo pone en su top `top_frac`, que % son realmente exitos
      (top `hit_frac` real). Sin modelo (al azar) seria `hit_frac`.
    - lift_top: cuantas veces mejor que el azar es esa seleccion.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_top = max(1, int(round(len(y_true) * top_frac)))
    top = np.argsort(-y_pred, kind="stable")[:n_top]
    is_hit = y_true >= np.quantile(y_true, 1 - hit_frac)
    precision = float(is_hit[top].mean())
    return {
        "pct_dentro_10": pct_within(y_true, y_pred, tolerance),
        "precision_top": precision * 100,
        "lift_top": precision / hit_frac,
    }
