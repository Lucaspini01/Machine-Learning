# src/models.py
from __future__ import annotations
import numpy as np
from typing import Optional

__all__ = ["KMeans", "GMM"]

# ---------- KMeans ----------
class KMeans:
    def __init__(self, n_clusters=8, n_init=5, max_iter=300, tol=1e-4, random_state: int = 0):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centers(self, X, rng):
        idx = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[idx].copy()

    def _assign(self, X, C):
        d2 = ((X[:, None, :] - C[None, :, :])**2).sum(axis=2)
        labels = d2.argmin(axis=1)
        inertia = d2[np.arange(X.shape[0]), labels].sum()
        return labels, inertia

    def _update(self, X, labels):
        centers = []
        for k in range(self.n_clusters):
            mask = (labels == k)
            centers.append(X[mask].mean(axis=0) if mask.any() else X[np.random.randint(0, X.shape[0])])
        return np.vstack(centers)

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        best_inertia = np.inf
        best_labels = None
        best_centers = None

        for _ in range(self.n_init):
            C = self._init_centers(X, rng)
            prev = np.inf
            for _it in range(self.max_iter):
                labels, inertia = self._assign(X, C)
                C = self._update(X, labels)
                if abs(prev - inertia) <= self.tol:
                    break
                prev = inertia
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = C

        self.inertia_ = best_inertia
        self.labels_ = best_labels
        self.cluster_centers_ = best_centers
        return self

    def predict(self, X):
        d2 = ((X[:, None, :] - self.cluster_centers_[None, :, :])**2).sum(axis=2)
        return d2.argmin(axis=1)

# ---------- GMM robusto ----------
class GMM:
    """
    GMM con:
      - cov_type: 'diag' (por defecto) o 'full'
      - reg_covar para estabilizar (default 1e-2)
      - init_means opcional (ideal: centroides de k-means)
      - tie-break leve en predict para evitar empates numéricos
    """
    def __init__(
        self,
        n_components: int = 8,
        max_iter: int = 200,
        tol: float = 1e-4,
        reg_covar: float = 1e-2,
        random_state: int = 0,
        init_means: Optional[np.ndarray] = None,
        cov_type: str = "diag",
    ):
        assert cov_type in ("diag", "full")
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg_covar
        self.rs = random_state
        self.init_means = init_means
        self.cov_type = cov_type

        self.pi_ = None
        self.mu_ = None
        self.Sigma_ = None
        self.resp_ = None
        self.loglik_ = None

    # ---- internals ----
    def _init(self, X):
        rng = np.random.default_rng(self.rs)
        N, D = X.shape
        if self.init_means is None:
            idx = rng.choice(N, self.K, replace=False)
            self.mu_ = X[idx].copy()
        else:
            self.mu_ = self.init_means.copy()

        base = np.cov(X.T) + self.reg * np.eye(D)
        if self.cov_type == "diag":
            base = np.diag(np.diag(base))
        self.Sigma_ = np.stack([base.copy() for _ in range(self.K)], axis=0)
        self.pi_ = np.ones(self.K) / self.K


    def _logpdf(self, X, mu, Sig):
        """
        Log densidad gaussiana con Cholesky (estable numéricamente).
        Devuelve un vector (N,) con log N(x|mu,Sig).
        """
        D = X.shape[1]
        chol = np.linalg.cholesky(Sig)
        # resolver (chol)^{-1} (X-mu)^T
        diff = (X - mu).T  # (D,N)
        sol  = np.linalg.solve(chol, diff)  # (D,N)
        quad = (sol**2).sum(axis=0)         # (N,)
        logdet = 2.0 * np.log(np.diag(chol)).sum()
        return -0.5 * (D*np.log(2.0*np.pi) + logdet + quad)

    def _e(self, X):
        """
        E-step en log-espacio:
        log w_{nk} = log pi_k + log N(x_n | mu_k, Sigma_k)
        r_{nk} = softmax_k(log w_{nk})
        """
        N, D = X.shape
        logW = np.empty((N, self.K), dtype=float)
        for k in range(self.K):
            logW[:, k] = np.log(self.pi_[k] + 1e-16) + self._logpdf(X, self.mu_[k], self.Sigma_[k])
        # log-sum-exp
        m = np.max(logW, axis=1, keepdims=True)
        logsumexp = m + np.log(np.exp(logW - m).sum(axis=1, keepdims=True))
        resp = np.exp(logW - logsumexp)
        loglik = float(logsumexp.sum())
        return resp, loglik

    def _m(self, X, R):
        N, D = X.shape
        Nk = R.sum(axis=0) + 1e-12
        self.pi_ = Nk / N
        self.mu_ = (R.T @ X) / Nk[:, None]
        self.Sigma_ = np.zeros((self.K, D, D))
        for k in range(self.K):
            Xc = X - self.mu_[k]
            Sk = (Xc.T * R[:, k]) @ Xc / Nk[k]
            if self.cov_type == "diag":
                Sk = np.diag(np.diag(Sk))
            Sk.flat[::D + 1] += self.reg
            self.Sigma_[k] = Sk

    # ---- API ----
    def fit(self, X):
        self._init(X)
        prev = -np.inf
        for _ in range(self.max_iter):
            R, ll = self._e(X)
            self._m(X, R)
            if abs(ll - prev) <= self.tol:
                break
            prev = ll
        self.resp_ = R
        self.loglik_ = ll
        return self

    def predict_proba(self, X):
        R, _ = self._e(X)
        return R

    def predict(self, X):
        rng = np.random.default_rng(self.rs)
        R = self.predict_proba(X)
        R = R + 1e-12 * rng.standard_normal(R.shape)  # tie-break
        return R.argmax(axis=1)
