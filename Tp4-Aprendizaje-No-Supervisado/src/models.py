 
import numpy as np
from typing import Optional, Tuple, Dict
import src.metrics as metrics

# ----------------------------
# K-Means (Punto 3.a)
# ----------------------------
class KMeans:
    def __init__(self, n_clusters=8, n_init=5, max_iter=300, tol=1e-4, random_state=0):
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

    def _e_step(self, X, centers):
        # distancias cuadráticas
        d2 = ((X[:, None, :] - centers[None, :, :])**2).sum(axis=2)  # (N, K)
        labels = d2.argmin(axis=1)
        inertia = d2[np.arange(X.shape[0]), labels].sum()
        return labels, inertia

    def _m_step(self, X, labels):
        centers = []
        for k in range(self.n_clusters):
            mask = labels == k
            if not np.any(mask):
                # re-inicializar un centro si quedó vacío
                centers.append(X[np.random.randint(0, X.shape[0])])
            else:
                centers.append(X[mask].mean(axis=0))
        return np.vstack(centers)

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        best_inertia, best_centers, best_labels = np.inf, None, None
        for _ in range(self.n_init):
            centers = self._init_centers(X, rng)
            prev_inertia = np.inf
            for _it in range(self.max_iter):
                labels, inertia = self._e_step(X, centers)
                centers = self._m_step(X, labels)
                if abs(prev_inertia - inertia) <= self.tol:
                    break
                prev_inertia = inertia
            if inertia < best_inertia:
                best_inertia, best_centers, best_labels = inertia, centers, labels
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        d2 = ((X[:, None, :] - self.cluster_centers_[None, :, :])**2).sum(axis=2)
        return d2.argmin(axis=1)

# ----------------------------
# Gaussian Mixture Model (EM) (Punto 3.b)
# ----------------------------
class GMM:
    def __init__(self, n_components=8, max_iter=200, tol=1e-4, reg_covar=1e-6, random_state=0, init_means: Optional[np.ndarray]=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.init_means = init_means
        # parámetros
        self.pi_ = None
        self.mu_ = None
        self.Sigma_ = None
        self.resp_ = None
        self.loglik_ = None

    def _init_params(self, X):
        rng = np.random.default_rng(self.random_state)
        N, D = X.shape
        if self.init_means is None:
            idx = rng.choice(N, self.n_components, replace=False)
            self.mu_ = X[idx].copy()
        else:
            self.mu_ = self.init_means.copy()
        self.Sigma_ = np.stack([np.cov(X.T) + self.reg_covar*np.eye(D) for _ in range(self.n_components)], axis=0)
        self.pi_ = np.ones(self.n_components) / self.n_components

    def _gaussian_pdf(self, X, mu, Sigma):
        D = X.shape[1]
        chol = np.linalg.cholesky(Sigma)
        solve = np.linalg.solve(chol, (X - mu).T)
        quad = (solve**2).sum(axis=0)
        log_det = 2*np.log(np.diag(chol)).sum()
        log_prob = -0.5*(D*np.log(2*np.pi) + log_det + quad)
        return np.exp(log_prob)

    def _e_step(self, X):
        N = X.shape[0]
        W = np.zeros((N, self.n_components))
        for k in range(self.n_components):
            W[:, k] = self.pi_[k] * self._gaussian_pdf(X, self.mu_[k], self.Sigma_[k])
        W_sum = W.sum(axis=1, keepdims=True) + 1e-12
        resp = W / W_sum
        loglik = np.log(W_sum).sum()
        return resp, loglik

    def _m_step(self, X, resp):
        N, D = X.shape
        Nk = resp.sum(axis=0) + 1e-12
        self.pi_ = Nk / N
        self.mu_ = (resp.T @ X) / Nk[:, None]
        self.Sigma_ = np.zeros((self.n_components, D, D))
        for k in range(self.n_components):
            Xc = X - self.mu_[k]
            self.Sigma_[k] = (Xc.T * resp[:, k]) @ Xc / Nk[k]
            self.Sigma_[k].flat[::D+1] += self.reg_covar  # regularización diag

    def fit(self, X):
        self._init_params(X)
        prev = -np.inf
        for _ in range(self.max_iter):
            resp, loglik = self._e_step(X)
            self._m_step(X, resp)
            if abs(loglik - prev) <= self.tol:
                break
            prev = loglik
        self.resp_ = resp
        self.loglik_ = loglik
        return self

    def predict_proba(self, X):
        resp, _ = self._e_step(X)
        return resp

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

def gmm_num_params(D, K, cov_type="full"):
    """
    Nº de parámetros de un GMM con covarianzas 'full':
    - pesos: K-1 (uno se determina por normalización)
    - medias: K*D
    - covarianzas: K * D*(D+1)/2
    """
    if cov_type != "full":
        raise NotImplementedError("Solo 'full' implementado aquí.")
    return (K - 1) + K * D + K * (D * (D + 1) // 2)

def fit_best_gmm(Z, K, n_init=3, reg_covar=1e-6, random_state=0, use_kmeans_init=True):
    """
    Corre GMM varias veces y devuelve el mejor por log-likelihood.
    Opcionalmente inicializa medias con KMeans para estabilizar.
    """
    best = {"model": None, "loglik": -np.inf, "labels": None}
    rng = np.random.default_rng(random_state)
    init_means_list = [None] * n_init

    if use_kmeans_init:
        km = KMeans(n_clusters=K, n_init=5, random_state=random_state).fit(Z)
        init_means_list[0] = km.cluster_centers_
        # el resto al azar
        for i in range(1, n_init):
            idx = rng.choice(Z.shape[0], size=K, replace=False)
            init_means_list[i] = Z[idx].copy()
    else:
        for i in range(n_init):
            idx = rng.choice(Z.shape[0], size=K, replace=False)
            init_means_list[i] = Z[idx].copy()

    for init_means in init_means_list:
        gmm = GMM(n_components=K, reg_covar=reg_covar, random_state=random_state, init_means=init_means).fit(Z)
        if gmm.loglik_ > best["loglik"]:
            best.update({"model": gmm, "loglik": gmm.loglik_, "labels": gmm.predict(Z)})
    return best

def gmm_sweep_over_K(Z, Ks, n_init=3, random_state=0):
    D = Z.shape[1]
    logliks, bics, aics, sils = [], [], [], []

    for K in Ks:
        best = fit_best_gmm(Z, K, n_init=n_init, reg_covar=1e-3 ,random_state=random_state,use_kmeans_init=True)
        p = gmm_num_params(D, K, "full")
        N = Z.shape[0]
        ll = best["loglik"]
        aic = -2 * ll + 2 * p
        bic = -2 * ll + p * np.log(N)

        labels = best["labels"]
        uniq, counts = np.unique(labels, return_counts=True)
        if len(uniq) < 2 or np.any(counts < 2):
            sil = np.nan
        else:
            sil = metrics.silhouette_score(Z, labels)

        logliks.append(ll); aics.append(aic); bics.append(bic); sils.append(sil)

    return {
        "Ks": np.array(Ks),
        "loglik": np.array(logliks),
        "AIC": np.array(aics),
        "BIC": np.array(bics),
        "silhouette": np.array(sils),
    }
