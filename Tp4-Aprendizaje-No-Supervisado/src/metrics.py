import numpy as np

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette promedio para etiquetas dadas.
    Robusta a:
      - X con shape (N,)  -> se trata como (N,1)
      - clusters singulares (tamaño 1) -> S(i)=0
      - casos degenerados (un solo cluster) -> np.nan
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    # Aceptar latentes 1D como (N,1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    N = X.shape[0]
    uniq = np.unique(labels)
    if uniq.size < 2:
        return np.nan  # silhouette no definido con <2 clusters

    # Distancias euclídeas pareadas (N,N)
    dists = np.sqrt(((X[:, None, :] - X[None, :, :])**2).sum(axis=2))

    S = np.zeros(N, dtype=np.float64)
    for i in range(N):
        li = labels[i]
        same = (labels == li)
        other = ~same

        # a(i): promedio intra-cluster excluyendo i
        # si el cluster es singleton, a=0 y S(i) se decidirá con b
        if same.sum() > 1:
            ai_vals = dists[i, same]
            ai_vals = ai_vals[ai_vals > 0]  # excluye distancia a sí mismo
            a = ai_vals.mean() if ai_vals.size else 0.0
        else:
            a = 0.0

        # b(i): mínimo promedio de distancia a otro cluster
        b = np.inf
        for c in uniq:
            if c == li:
                continue
            mask = (labels == c)
            if mask.any():
                b = min(b, dists[i, mask].mean())

        if not np.isfinite(b):
            S[i] = 0.0  # degenerado: no hay otro cluster con muestras
            continue

        denom = max(a, b)
        S[i] = 0.0 if denom == 0.0 else (b - a) / denom

    return float(np.nanmean(S))



def elbow_gains(inertias: np.ndarray) -> np.ndarray:
    """
    Ganancia marginal de la inercia entre K y K-1 (para el elbow del 3.c).
    """
    inertias = np.asarray(inertias, dtype=np.float64)
    gains = np.empty_like(inertias)
    gains[0] = np.nan
    gains[1:] = inertias[:-1] - inertias[1:]
    return gains
