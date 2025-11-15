import numpy as np

def split_dataset(X, y, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42):
    np.random.seed(seed)
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    n_train = int(frac_train * n)
    n_val = int(frac_val * n)
    n_test = n - n_train - n_val

    idx_train = indices[:n_train]
    idx_val   = indices[n_train:n_train + n_val]
    idx_test  = indices[n_train + n_val:]

    X_train, y_train = X[idx_train], y[idx_train]
    X_val,   y_val   = X[idx_val],   y[idx_val]
    X_test,  y_test  = X[idx_test],  y[idx_test]

    return X_train, y_train, X_val, y_val, X_test, y_test

# One-hot encoding
def one_hot(y, num_classes):
    return np.eye(num_classes)[y].T 

import numpy as np

def undersample_partial(X, y, factor=0.5, random_state=42):
    """
    Realiza un undersampling parcial reduciendo las clases más grandes
    sin igualarlas completamente a la clase más pequeña.

    Parámetros
    ----------
    X : np.ndarray
        Datos de entrada de forma (N, ...).
    y : np.ndarray
        Etiquetas correspondientes de forma (N,).
    factor : float
        Nivel de reducción de las clases grandes.
        1.0 = sin recorte, 0.0 = igualar todas a la clase menor.
        Ejemplo: factor=0.5 reduce las clases grandes a un punto medio entre
        la clase menor y la mayor.
    random_state : int
        Semilla para reproducibilidad.

    Retorna
    -------
    X_res, y_res : np.ndarray
        Dataset parcialmente balanceado.
    """
    np.random.seed(random_state)
    classes, counts = np.unique(y, return_counts=True)
    n_min, n_max = np.min(counts), np.max(counts)
    n_target = int(n_min + factor * (n_max - n_min))

    print(f"Tamaño objetivo aproximado por clase: {n_target}")

    X_resampled = []
    y_resampled = []

    for cls, count in zip(classes, counts):
        X_cls = X[y == cls]
        y_cls = y[y == cls]

        # si hay más muestras que el tamaño objetivo, recortar aleatoriamente
        if count > n_target:
            indices = np.random.choice(count, size=n_target, replace=False)
            X_sel = X_cls[indices]
            y_sel = y_cls[indices]
        else:
            X_sel = X_cls
            y_sel = y_cls

        X_resampled.append(X_sel)
        y_resampled.append(y_sel)

    X_resampled = np.concatenate(X_resampled, axis=0)
    y_resampled = np.concatenate(y_resampled, axis=0)

    print("Distribución de clases luego del undersampling parcial:")
    for cls in classes:
        print(f"Clase {cls:2d}: {(y_resampled == cls).sum()} muestras")

    return X_resampled, y_resampled

