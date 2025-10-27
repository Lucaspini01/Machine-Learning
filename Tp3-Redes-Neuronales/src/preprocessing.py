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

# --- One-hot encoding ---
def one_hot(y, num_classes):
    return np.eye(num_classes)[y].T  # salida (num_classes, N)

