import numpy as np

import numpy as np

class LogisticRegressionL2:
    def __init__(self, L2=0.0, learning_rate=0.01, n_iter=1000):
        self.L2 = float(L2)
        self.learning_rate = float(learning_rate)
        self.n_iter = int(n_iter)
        self.weights = None
        self.bias = 0.0

    def _check_fitted(self):
        if self.weights is None:
            raise RuntimeError("El modelo no está entrenado: self.weights es None. Llamá a fit(X, y) primero.")

    def sigmoid(self, z):
        z = np.asarray(z, dtype=np.float64)
        z = np.clip(z, -35.0, 35.0)  # evita overflow en exp
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        # Inicialización
        if self.weights is None:
            self.weights = np.zeros(d, dtype=np.float64)
        if self.bias is None:
            self.bias = 0.0

        # GD con weight decay (equivalente a L2, no regulariza bias)
        lr = self.learning_rate
        lam = self.L2

        for _ in range(self.n_iter):
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            grad_w = (X.T @ (y_pred - y)) / n
            grad_b = np.sum(y_pred - y) / n

            self.weights = (1.0 - lr * lam) * self.weights - lr * grad_w
            self.bias    = self.bias - lr * grad_b
        return self

    def predict_proba(self, X):
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
