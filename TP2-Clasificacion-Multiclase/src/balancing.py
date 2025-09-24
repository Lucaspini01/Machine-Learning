import numpy as np
import pandas as pd

# Utilidades de clase

def class_counts(y, positive_label=None):
    y = np.asarray(y)
    vals, cnts = np.unique(y, return_counts=True)
    counts = dict(zip(vals, cnts))
    if positive_label is not None and positive_label not in counts:
        counts[positive_label] = 0
    return counts

def minority_majority_labels(y):
    counts = class_counts(y)
    # retorna (minority_label, majority_label)
    items = sorted(counts.items(), key=lambda kv: kv[1])
    return items[0][0], items[-1][0]

def class_priors(y):
    y = np.asarray(y)
    counts = class_counts(y)
    n = len(y)
    return {c: counts[c] / n for c in counts}


# SIN REBALANCEO
def no_rebalance(df, target):
    """Devuelve el mismo DF (copia)"""
    return df.copy()


# UNDERSAMPLING mayoritaria

def undersample_majority(df, target, random_state=42):
    rng = np.random.default_rng(random_state)
    y = df[target].values
    minor, major = minority_majority_labels(y)
    n_minor = (y == minor).sum()
    # tomar n_minor de la mayoritaria
    idx_major = np.where(y == major)[0]
    idx_minor = np.where(y == minor)[0]
    pick_major = rng.choice(idx_major, size=n_minor, replace=False)
    idx_final = np.concatenate([idx_minor, pick_major])
    rng.shuffle(idx_final)
    return df.iloc[idx_final].reset_index(drop=True)


# OVERSAMPLING duplicación

def oversample_duplicate(df, target, random_state=42):
    rng = np.random.default_rng(random_state)
    y = df[target].values
    minor, major = minority_majority_labels(y)
    n_major = (y == major).sum()
    idx_minor = np.where(y == minor)[0]
    idx_major = np.where(y == major)[0]
    # samplear con reemplazo desde la minoritaria hasta igualar la mayoritaria
    extra = n_major - len(idx_minor)
    add_idx = rng.choice(idx_minor, size=extra, replace=True) if extra > 0 else np.array([], dtype=int)
    idx_final = np.concatenate([idx_major, idx_minor, add_idx])
    rng.shuffle(idx_final)
    return df.iloc[idx_final].reset_index(drop=True)


# SMOTE sencillo

def smote_oversample(df, target, numeric_cols, k=5, random_state=42):
    """
    Genera muestras sintéticas de la clase minoritaria hasta igualar la mayoritaria.
    - numeric_cols: columnas numéricas (idealmente ya normalizadas).
    - Las columnas no numéricas se copian desde el punto base (minor).
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()

    y = df[target].values
    minor, major = minority_majority_labels(y)
    df_minor = df[df[target] == minor].reset_index(drop=True)
    df_major = df[df[target] == major].reset_index(drop=True)

    n_minor, n_major = len(df_minor), len(df_major)
    if n_minor >= n_major:
        return out 

    # Matriz de características minoritarias (solo numéricas)
    X = df_minor[numeric_cols].values.astype(float)
    n = X.shape[0]

    if n < 2:
        # si solo hay 1 minor, no se puede SMOTE: duplicamos
        need = n_major - n_minor
        dup_idx = rng.choice(np.arange(n), size=need, replace=True)
        synth = df_minor.iloc[dup_idx].copy()
        out = pd.concat([df, synth], axis=0, ignore_index=True)
        return out

    # Pre-computar KNN en la minoritaria
    # Distancias euclídeas pairwise (O(n^2), suficiente para tamaños medianos)
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    # vecinos por fila (excluyendo el mismo índice)
    knn_idx = np.argsort(dists, axis=1)[:, 1:k+1]  # los k más cercanos

    need = n_major - n_minor
    synth_rows = []
    for _ in range(need):
        i = int(rng.integers(0, n))
        j = int(knn_idx[i, rng.integers(0, min(k, knn_idx.shape[1]))])
        alpha = rng.random() 
        x_i = X[i]
        x_j = X[j]
        x_syn = x_i + alpha * (x_j - x_i)

        new_row = df_minor.iloc[i].copy()
        new_row.loc[numeric_cols] = x_syn
        new_row[target] = minor
        synth_rows.append(new_row)

    synth_df = pd.DataFrame(synth_rows, columns=df.columns)
    out = pd.concat([df, synth_df], axis=0, ignore_index=True)
    return out


# COST RE-WEIGHTING

def cost_reweighting_sample_weights(y):
    """
    Devuelve sample_weight para reponderar la clase minoritaria con:
      C = pi_major / pi_minor
    """
    y = np.asarray(y)
    priors = class_priors(y)
    minor, major = minority_majority_labels(y)
    pi1 = priors[minor]
    pi2 = priors[major]
    C = pi2 / pi1 if pi1 > 0 else 1.0
    w = np.ones_like(y, dtype=float)
    w[y == minor] = C
    return w, {"minor": minor, "major": major, "pi_minor": pi1, "pi_major": pi2, "C": C}
