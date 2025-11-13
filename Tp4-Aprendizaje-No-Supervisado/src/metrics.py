# src/metrics.py
import numpy as np

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette sin scikit-learn. O(N^2) en tiempo.
    S(i) = (b(i) - a(i)) / max(a(i), b(i))
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    N = X.shape[0]
    # distancias pares
    dists = np.sqrt(((X[:, None, :] - X[None, :, :])**2).sum(axis=2))  # (N,N)
    S = np.zeros(N, dtype=np.float64)
    for i in range(N):
        same = labels == labels[i]
        other = ~same
        a = dists[i, same]
        a = a[a > 0].mean() if np.any(a > 0) else 0.0
        # promedio mínimo a clúster distinto
        b = np.inf
        for c in np.unique(labels[other]):
            mask = labels == c
            b = min(b, dists[i, mask].mean())
        S[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(S.mean())

def elbow_gains(inertias: np.ndarray) -> np.ndarray:
    """
    Ganancia marginal de la inercia entre K y K-1 (para el elbow del 3.c).
    """
    inertias = np.asarray(inertias, dtype=np.float64)
    gains = np.empty_like(inertias)
    gains[0] = np.nan
    gains[1:] = inertias[:-1] - inertias[1:]
    return gains
