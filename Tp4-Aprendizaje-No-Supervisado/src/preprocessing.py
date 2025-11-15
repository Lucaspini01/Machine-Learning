# src/preprocessing.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

# ----------------------------
# Carga y utilidades de imágenes (Punto 1.a y 1.b)
# ----------------------------
def load_faces_csv(path: str, label_col: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Carga caras.csv.
    - Si label_col se pasa (e.g., "person_id"), la usa como etiqueta.
    - Si label_col es None, intenta detectar 'person_id' o 'label'.
    - Si no hay etiqueta detectable, devuelve y=None.
    Devuelve:
      X: (N, D) float32 con pixeles en [0,1] si venían 0..255
      y: (N,) con etiquetas, o None si no hay
    """
    df = pd.read_csv(path)

    # Detección de columna de etiqueta si no se especifica
    detected = None
    if label_col is None:
        for cand in ("person_id", "label"):
            if cand in df.columns:
                detected = cand
                break
    else:
        detected = label_col if label_col in df.columns else None

    if detected is None:
        X = df.to_numpy(dtype=np.float32)
        y = None
    else:
        y = df[detected].to_numpy()
        X = df.drop(columns=[detected]).to_numpy(dtype=np.float32)

    # Normalización a [0,1] si corresponde
    if X.size > 0 and X.max() > 1.0:
        X = X / 255.0

    return X, y

def plot_images(X: np.ndarray, idxs: np.ndarray, img_shape=(64,64), ncols=5, suptitle=None):
    """
    Grafica un conjunto arbitrario de imágenes (Punto 1.a).
    """
    n = len(idxs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.8*ncols, 1.8*nrows))
    axes = np.array(axes).reshape(-1)
    for ax, i in zip(axes, idxs):
        ax.imshow(X[i].reshape(img_shape), cmap="gray")
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    return fig

def plot_by_class(X, y, classes, per_class=5, img_shape=(64,64)):
    """
    Muestras agrupadas por clase (Punto 1.b).
    """
    idxs = []
    for c in classes:
        where = np.where(y == c)[0]
        take = np.random.choice(where, size=min(per_class, len(where)), replace=False)
        idxs.extend(take.tolist())
    idxs = np.array(idxs)
    return plot_images(X, idxs, img_shape=img_shape, suptitle=f"Muestras por clase {classes}")

def plot_class_distribution(y):
    """
    Gráfico de distribución de clases.
    """
    vals, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(vals.astype(str), counts)
    ax.set_xlabel("Clase")
    ax.set_ylabel("Cantidad de muestras")
    ax.set_title("Distribución de clases")
    return fig

# ----------------------------
# Split estratificado (Punto 1.c)
# ----------------------------
def stratified_split(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    if y is None:
        N = X.shape[0]
        idx = np.arange(N); rng.shuffle(idx)
        n_test = int(round(N * test_size))
        te, tr = idx[:n_test], idx[n_test:]
        return (X[tr], None), (X[te], None)
    # estratificado
    y = np.asarray(y)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        ii = np.where(y == cls)[0]
        rng.shuffle(ii)
        n_t = max(1, int(np.round(len(ii) * test_size)))
        test_idx.extend(ii[:n_t].tolist())
        train_idx.extend(ii[n_t:].tolist())
    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])


# ----------------------------
# Estandarización + PCA (Punto 2.a y 2.b)
# ----------------------------
def standardize_fit(X: np.ndarray) -> Dict[str, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    sigma[sigma == 0] = 1.0
    return {"mu": mu, "sigma": sigma}

def standardize_transform(X: np.ndarray, std: Dict[str, np.ndarray]) -> np.ndarray:
    return (X - std["mu"]) / std["sigma"]

def pca_fit(X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    PCA por SVD sobre datos centrados (se espera X ya estandarizado o al menos centrado).
    Devuelve componentes, valores singulares y varianza explicada.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt  # (D, D) filas = componentes
    explained_var = (S**2) / (X.shape[0] - 1)
    explained_ratio = explained_var / explained_var.sum()
    cum_explained = np.cumsum(explained_ratio)
    return {
        "mean": Xc.mean(axis=0, keepdims=True) * 0.0,  # ya centrado localmente
        "components": comps,
        "singular_values": S,
        "explained_variance": explained_var,
        "explained_ratio": explained_ratio,
        "cum_explained_ratio": cum_explained
    }

def pca_choose_k_from_variance(cum_explained_ratio: np.ndarray, thresh: float = 0.90) -> int:
    """
    Elegir k >= 1 tal que la varianza explicada acumulada >= thresh (Punto 2.b).
    """
    return int(np.searchsorted(cum_explained_ratio, thresh) + 1)

def pca_transform(X: np.ndarray, pca: Dict[str, np.ndarray], k: int) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    W = pca["components"][:k]  # (k, D)
    Z = Xc @ W.T               # (N, k)
    return Z

def pca_inverse_transform(Z: np.ndarray, pca: Dict[str, np.ndarray], k: int, X_mean: Optional[np.ndarray]=None) -> np.ndarray:
    W = pca["components"][:k]  # (k, D)
    Xrec = Z @ W
    if X_mean is not None:
        Xrec = Xrec + X_mean
    return Xrec

def explained_variance_plot(pca: Dict[str, np.ndarray]):
    """
    Gráfico de varianza explicada acumulada (Punto 2.b).
    """
    cum = pca["cum_explained_ratio"]
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(np.arange(1, len(cum)+1), cum, marker="o")
    ax.axhline(0.90, linestyle="--")
    ax.set_xlabel("Número de componentes")
    ax.set_ylabel("Varianza explicada acumulada")
    ax.grid(True, alpha=0.3)
    return fig

def z_standardize(Z):
    """
    Estandariza Z a media 0 y desviación estándar 1 por característica.
    """
    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (Z - mu) / sd
