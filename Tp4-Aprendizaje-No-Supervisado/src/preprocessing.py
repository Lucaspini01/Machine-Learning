# src/preprocessing.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

__all__ = [
    "load_faces_csv",
    "stratified_split",
    "standardize_fit", "standardize_transform",
    "pca_fit", "pca_transform", "pca_inverse",
    "zstd_fit", "zstd_transform",
]

# ---------- Carga de datos ----------
def load_faces_csv(path: str, label_col: Optional[str] = "person_id") -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga un CSV con pixeles en columnas y una columna de etiqueta (por defecto 'person_id').
    Si label_col no existe, usa la última columna como etiqueta.
    Devuelve:
      X: (N, D) float32 con valores en [0,1] si venían 0..255
      y: (N,) etiquetas
    """
    df = pd.read_csv(path)
    if label_col not in df.columns:
        label_col = df.columns[-1]
    y = df[label_col].to_numpy()
    X = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    if X.size > 0 and X.max() > 1.0:
        X = X / 255.0
    return X, y

# ---------- Split estratificado ----------
def stratified_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 0):
    """
    Split estratificado sin scikit-learn.
    Devuelve: (X_train, y_train), (X_test, y_test)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])

# ---------- Estandarización ----------
def standardize_fit(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Calcula media y desvío por feature para estandarizar X."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return {"mu": mu, "sd": sd}

def standardize_transform(X: np.ndarray, st: Dict[str, np.ndarray]) -> np.ndarray:
    """Aplica estandarización usando parámetros de train."""
    return (X - st["mu"]) / st["sd"]

# ---------- PCA (con media guardada) ----------
def pca_fit(X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Ajusta PCA por SVD, guardando media de train.
    Devuelve:
      'mu': media (1,D)
      'components': Vt (D,D) — filas son componentes principales
      'cum_explained_ratio': varianza explicada acumulada
    """
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = (S**2) / (X.shape[0] - 1)
    er = ev / ev.sum()
    return {
        "mu": mu,
        "components": Vt,
        "cum_explained_ratio": np.cumsum(er),
    }

def pca_transform(X: np.ndarray, pca: Dict[str, np.ndarray], k: int) -> np.ndarray:
    """Transforma X al subespacio de las primeras k PCs usando la media de train (pca['mu'])."""
    Xc = X - pca["mu"]
    W = pca["components"][:k]          # (k, D)
    Z = Xc @ W.T                       # (N, k)
    return Z

def pca_inverse(Z: np.ndarray, pca: Dict[str, np.ndarray], k: int) -> np.ndarray:
    """Reconstruye desde Z usando las primeras k PCs y la media guardada."""
    W = pca["components"][:k]          # (k, D)
    Xrec = Z @ W + pca["mu"]
    return Xrec

# ---------- Estandarización de latentes (para GMM) ----------
def zstd_fit(Z: np.ndarray) -> Dict[str, np.ndarray]:
    """Media y desvío por dimensión del espacio latente."""
    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return {"mu": mu, "sd": sd}

def zstd_transform(Z: np.ndarray, st: Dict[str, np.ndarray]) -> np.ndarray:
    """Estandariza latentes usando parámetros de train."""
    return (Z - st["mu"]) / st["sd"]
