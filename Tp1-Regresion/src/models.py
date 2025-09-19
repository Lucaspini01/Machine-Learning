import numpy as np

class RegresionLineal:
    def __init__(self, X: np.ndarray, y: np.ndarray, estandarizar: bool = False, feature_names: list = None, L1=0 , L2=0):
        self.X = np.asarray(X, dtype=float) # Matriz de características
        self.y = np.asarray(y, dtype=float).reshape(-1, 1) # Target y
        self.n, self.d = self.X.shape # Dimensiones

        self.feature_names = feature_names or [f"x{i+1}" for i in range(self.d)] # Nombres de las características
        self.estandarizar = estandarizar # Indica si se debe estandarizar dentro del modelo
        self.mu_ = None  # Valor de mu al normalizar
        self.sigma_ = None # Valor de sigma al normalizar
        self.coef = None  # Aca voy a almacenar los valores de los coeficientes

    # Utilidades
    
    def _fit_standardizer(self, X):
        self.mu_ = X.mean(axis=0)
        self.sigma_ = X.std(axis=0, ddof=0)
        self.sigma_[self.sigma_ == 0.0] = 1.0  # evita división por 0

    def _transform(self, X):
        if not self.estandarizar:
            return X
        if self.mu_ is None or self.sigma_ is None:
            raise RuntimeError("Estandarizador no ajustado todavía.")
        return (X - self.mu_) / self.sigma_

    @staticmethod
    def _agregar_bias(X):
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([ones, X])

    def imprimir_coeficientes(self):
        if self.coef is None:
            print("El modelo no está entrenado todavía.")
            return
        print(f"  Bias: {self.coef[0]:.4f}")
        for name, w in zip(self.feature_names, self.coef[1:]):
            print(f"  {name}: {w:.4f}")

    # Entrenamiento por pseudoinversa 

    def entrenar_pseudoinversa(self, rcond=None):
        X = self.X.copy()
        Xb = self._agregar_bias(X)

        U, s, Vt = np.linalg.svd(Xb, full_matrices=False)
        if rcond is None:
            rcond = np.finfo(Xb.dtype).eps * max(Xb.shape)
        cutoff = rcond * s[0]
        s_inv = np.where(s > cutoff, 1.0 / s, 0.0)

        w = Vt.T @ (s_inv[:, None] * (U.T @ self.y))
        self.coef = w.flatten()
        return self

    # -Entrenamiento por gradiente descendente 

    def entrenar_gradiente_descendente(self, alpha=0.01, n_iter=1000):
        X = self.X.copy()

        Xb = self._agregar_bias(X)
        self.coef = np.zeros(Xb.shape[1])

        for _ in range(n_iter):
            pred = Xb @ self.coef
            error = pred - self.y.ravel()
            gradient = Xb.T @ error / Xb.shape[0]
            self.coef -= alpha * gradient

        return self
    
    def entrenar_ridge(self, lam: float = 0.0):
        """
        Ridge (L2) con solución cerrada:
        w = (Xb^T Xb + lam * D)^(-1) Xb^T y
        con D = diag(0, 1, 1, ..., 1) para NO penalizar el bias.
        """
        X = self.X.copy()
        Xb = self._agregar_bias(X)    
        n, p = Xb.shape

        # Matriz de penalización que no afecta el bias
        D = np.eye(p)
        D[0, 0] = 0.0

        A = Xb.T @ Xb + lam * D
        b = Xb.T @ self.y
        w = np.linalg.solve(A, b)             
        self.coef = w.flatten()
        return self

    def _lipschitz_MSE(self, Xb):
        # L = 2/n * s_max^2  (s_max = mayor singular de Xb)
        n = Xb.shape[0]
        # uso SVD compacto
        _, s, _ = np.linalg.svd(Xb, full_matrices=False)
        s_max = s[0] if s.size else 0.0
        return (2.0 / n) * (s_max ** 2)

    def entrenar_lasso(
        self,
        lam: float = 1e-3,
        lr: float = None,        
        max_iter: int = 10000,
        tol: float = 1e-8,
        verbose: bool = False
    ):
        X = self.X.copy()
        y = self.y.copy()
        # Chequeo de finitud
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise ValueError("X/y contienen NaN o Inf. Limpia antes de entrenar.")

        Xb = self._agregar_bias(X)  
        n, p = Xb.shape

        if lr is None:
            L = self._lipschitz_MSE(Xb)
            # factor de seguridad
            lr = 0.9 / (L + 1e-12)

        w = np.zeros((p, 1))
        prev_loss = np.inf

        def soft_threshold(w_vec, thr):
            out = w_vec.copy()
            z = out[1:, 0]
            out[1:, 0] = np.sign(z) * np.maximum(np.abs(z) - thr, 0.0)
            return out

        for it in range(max_iter):
            # gradiente smooth
            resid = Xb @ w - y
            grad = (2.0 / n) * (Xb.T @ resid)
            w_tmp = w - lr * grad
            w_new = soft_threshold(w_tmp, lr * lam)

            # recomputo residuo para la loss con w_new
            resid_new = Xb @ w_new - y
            loss = float((resid_new ** 2).mean() + lam * np.sum(np.abs(w_new[1:, 0])))

            # chequeo que sea finito
            if not np.isfinite(loss):
                if verbose:
                    print("Loss no finita: reduce lr o revisa X/y.")

                lr *= 0.5
                continue

            if abs(prev_loss - loss) < tol:
                if verbose:
                    print(f"LASSO convergió en {it} it. Loss={loss:.6f}, lr={lr:.3e}")
                w = w_new
                break

            if loss > prev_loss * 1.001:
                lr *= 0.5 
                continue

            w = w_new
            prev_loss = loss
        else:
            if verbose:
                print(f"LASSO terminó en max_iter={max_iter}. Loss={prev_loss:.6f}, lr={lr:.3e}")

        self.coef = w.flatten()
        return self


    # Predicción
    def predecir(self, X_new: np.ndarray) -> np.ndarray:
        X_new = np.asarray(X_new, dtype=float)
        X_new = self._transform(X_new) if self.estandarizar else X_new
        Xb_new = self._agregar_bias(X_new)
        if self.coef is None:
            raise RuntimeError("El modelo no está entrenado.")
        return (Xb_new @ self.coef.reshape(-1, 1)).ravel()