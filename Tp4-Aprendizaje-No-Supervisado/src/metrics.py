# src/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Tuple, Optional

__all__ = [
    "silhouette_score", "silhouette_score_robust",
    "elbow_gains", "select_k_by_elbow",
    "gmm_num_params", "purity_weighted_mean",
]

# ---------- Silhouette robusto ----------
def silhouette_score_robust(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette promedio robusto:
      - Acepta X con shape (N,) -> lo trata como (N,1)
      - Si hay <2 clusters o algún cluster singleton, devuelve np.nan
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    uniq, counts = np.unique(labels, return_counts=True)
    if uniq.size < 2 or (counts < 2).any():
        return np.nan

    # Distancias euclídeas pareadas
    dists = np.sqrt(((X[:, None, :] - X[None, :, :])**2).sum(axis=2))
    N = X.shape[0]
    S = np.zeros(N, dtype=float)

    for i in range(N):
        li = labels[i]
        same = (labels == li)
        other = ~same

        # a(i): intra-cluster (excluyendo i)
        ai = dists[i, same]
        ai = ai[ai > 0]
        a = ai.mean() if ai.size else 0.0

        # b(i): mínimo promedio hacia otro cluster
        b = np.inf
        for c in uniq:
            if c == li:
                continue
            mask = (labels == c)
            if mask.any():
                b = min(b, dists[i, mask].mean())

        if not np.isfinite(b):
            S[i] = 0.0
        else:
            den = max(a, b)
            S[i] = 0.0 if den == 0.0 else (b - a) / den

    return float(np.nanmean(S))

# Alias conveniente
def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    return silhouette_score_robust(X, labels)

# ---------- Elbow / ganancias decrecientes ----------
def elbow_gains(inertias: np.ndarray) -> np.ndarray:
    """
    Ganancia marginal de inercia: ΔJ(K) = J(K-1) - J(K)
    Devuelve un array alineado con J; gains[0] = np.nan.
    """
    J = np.asarray(inertias, dtype=float)
    gains = np.full_like(J, np.nan, dtype=float)
    gains[1:] = J[:-1] - J[1:]
    return gains

def select_k_by_elbow(inertias: Iterable[float], Ks: Iterable[int], rel_thresh: float = 0.05):
    """
    Selecciona K por:
      - Umbral relativo de ganancia: primer K con ΔJ/J(K-1) < rel_thresh
      - Segunda diferencia (codo): K con máxima caída de la ganancia
    Devuelve: (k_rel, k_dd, diag_dict_con_J_d_r_dd)
    """
    Ks = np.array(list(Ks), dtype=int)
    J  = np.array(list(inertias), dtype=float)

    d  = np.full_like(J, np.nan); d[1:]  = J[:-1] - J[1:]
    r  = np.full_like(J, np.nan); r[1:]  = d[1:] / J[:-1]
    dd = np.full_like(J, np.nan); dd[2:] = d[1:-1] - d[2:]

    cand = np.where(r < rel_thresh)[0]
    k_rel = Ks[cand[0]] if cand.size else Ks[np.nanargmax(r)]
    k_dd  = Ks[np.nanargmax(dd)]

    diag = {"J": J, "d": d, "r": r, "dd": dd}
    return int(k_rel), int(k_dd), diag

# ---------- Parámetros del GMM (para AIC/BIC) ----------
def gmm_num_params(D: int, K: int, cov_type: str = "full") -> int:
    """
    Nº de parámetros de un GMM:
      - pesos: K-1
      - medias: K*D
      - covarianzas:
          full -> K * D*(D+1)/2
          diag -> K * D
    """
    if cov_type == "full":
        cov = K * (D * (D + 1) // 2)
    elif cov_type == "diag":
        cov = K * D
    else:
        raise ValueError("cov_type debe ser 'full' o 'diag'")
    return (K - 1) + K * D + cov

# ---------- Pureza ----------
def purity_weighted_mean(labels: np.ndarray, y_true: np.ndarray) -> Tuple[float, pd.DataFrame, pd.Series]:
    """
    Devuelve:
      - pureza ponderada global
      - crosstab (clusters x clases)
      - pureza por cluster (serie)
    """
    labels = pd.Series(labels, name="cluster")
    classes = pd.Series(y_true, name="class")
    tab = pd.crosstab(labels, classes)
    purity = (tab.max(axis=1) / tab.sum(axis=1))
    pwm = float((purity * tab.sum(axis=1) / tab.sum().sum()).sum())
    return pwm, tab, purity
