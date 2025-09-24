import numpy as np
from collections import Counter

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

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        # Inicialización
        if self.weights is None:
            self.weights = np.zeros(d, dtype=np.float64)
        if self.bias is None:
            self.bias = 0.0

        # pesos de cada muestra
        if sample_weight is None:
            sample_weight = np.ones(n, dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        lr = self.learning_rate
        lam = self.L2

        for _ in range(self.n_iter):
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            # residuales ponderados
            residual = (y_pred - y) * sample_weight

            # gradientes ponderados
            grad_w = (X.T @ residual) / np.sum(sample_weight)
            grad_b = np.sum(residual) / np.sum(sample_weight)

            # actualización con weight decay (L2 solo en weights, no bias)
            self.weights = (1.0 - lr * lam) * self.weights - lr * grad_w
            self.bias = self.bias - lr * grad_b

        return self


    def predict_proba(self, X):
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

class LDA:
    def __init__(self):
        self.means_ = {}
        self.priors_ = {}
        self.cov_ = None
        self.inv_cov_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        n, d = X.shape
        self.cov_ = np.zeros((d, d))
        self.means_ = {}
        self.priors_ = {}

        for c in self.classes_:
            Xc = X[y == c]
            self.means_[c] = Xc.mean(axis=0)
            self.priors_[c] = len(Xc) / n
            self.cov_ += np.cov(Xc, rowvar=False) * (len(Xc)-1)

        self.cov_ /= (n - len(self.classes_))
        self.inv_cov_ = np.linalg.pinv(self.cov_)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        scores = []
        for c in self.classes_:
            mean = self.means_[c]
            prior = self.priors_[c]
            # discriminant function: linear in X
            g = X @ (self.inv_cov_ @ mean) - 0.5 * mean.T @ self.inv_cov_ @ mean + np.log(prior)
            scores.append(g)
        scores = np.array(scores).T
        # softmax para probabilidades
        exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    def _entropy(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return -np.sum([p*np.log2(p) for p in probs if p > 0])

    def _best_split(self, X, y):
        n, d = X.shape
        best_feat, best_thr, best_gain = None, None, -1
        base_entropy = self._entropy(y)

        for feat in range(d):
            thresholds = np.unique(X[:, feat])
            for thr in thresholds:
                left = y[X[:, feat] <= thr]
                right = y[X[:, feat] > thr]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = base_entropy - (len(left)/n)*self._entropy(left) - (len(right)/n)*self._entropy(right)
                if gain > best_gain:
                    best_feat, best_thr, best_gain = feat, thr, gain
        return best_feat, best_thr

    def _build(self, X, y, depth):
        if len(np.unique(y)) == 1 or (self.max_depth and depth >= self.max_depth) or len(y) < self.min_samples_split:
            return Counter(y).most_common(1)[0][0]
        feat, thr = self._best_split(X, y)
        if feat is None:
            return Counter(y).most_common(1)[0][0]
        left_idx = X[:, feat] <= thr
        right_idx = ~left_idx
        return {
            "feat": feat,
            "thr": thr,
            "left": self._build(X[left_idx], y[left_idx], depth+1),
            "right": self._build(X[right_idx], y[right_idx], depth+1)
        }

    def fit(self, X, y):
        self.tree_ = self._build(np.asarray(X), np.asarray(y), depth=0)
        return self

    def _predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node["feat"]] <= node["thr"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in np.asarray(X)])


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        n, d = X.shape
        self.classes_ = np.unique(y)
        for _ in range(self.n_estimators):
            idx = np.random.choice(n, n, replace=True)
            feat_idx = np.random.choice(d, self.max_features or d, replace=False)
            Xb, yb = X[idx][:, feat_idx], y[idx]
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(Xb, yb)
            self.trees.append((tree, feat_idx))
        return self
    
    @staticmethod
    def _majority_vote(pred_matrix: np.ndarray) -> np.ndarray:
        """
        Aplica voto mayoritario a una matriz de predicciones.
        - pred_matrix: shape (n_trees, n_samples)
        - Devuelve: array de etiquetas (n_samples,)
        """
        n_samples = pred_matrix.shape[1]
        votos_finales = []
        for i in range(n_samples):
            votos_i = pred_matrix[:, i]
            ganador = Counter(votos_i).most_common(1)[0][0]
            votos_finales.append(ganador)
        return np.array(votos_finales)

    def predict(self, X):
        X = np.asarray(X)
        if not self.trees:
            raise RuntimeError("El bosque no está entrenado. Llamá a fit(X, y) primero.")
        preds = []
        for tree, feat_idx in self.trees:
            preds.append(tree.predict(X[:, feat_idx]))
        pred_mat = np.vstack(preds)  # (n_trees, n_samples)
        return self._majority_vote(pred_mat)

    def predict_proba(self, X):
        """
        Devuelve matriz (n_samples, n_classes) con las probabilidades
        estimadas por voto de los árboles.
        """
        X = np.asarray(X)
        if not self.trees:
            raise RuntimeError("El bosque no está entrenado. Llamá a fit(X, y) primero.")

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        class_to_idx = {c: j for j, c in enumerate(self.classes_)}

        # conteo de votos
        votos = np.zeros((n_samples, n_classes), dtype=float)
        for tree, feat_idx in self.trees:
            pred = tree.predict(X[:, feat_idx])
            for i, c in enumerate(pred):
                votos[i, class_to_idx[c]] += 1

        # normalización a probabilidades
        return votos / len(self.trees)
    
class LogisticRegressionMulticlass:
    def __init__(self, L2=0.0, learning_rate=0.01, n_iter=1000):
        self.L2 = L2
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.models = {}   # dict clase -> modelo binario

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        for c in classes:
            y_bin = (y == c).astype(float)
            model = LogisticRegressionL2(L2=self.L2,
                                         learning_rate=self.learning_rate,
                                         n_iter=self.n_iter)
            model.fit(X, y_bin)
            self.models[c] = model
        return self
    
    def predict_proba(self, X):
        """
        Devuelve matriz (n_samples, n_classes) con probas normalizadas.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probas = np.zeros((n_samples, n_classes))

        for j, c in enumerate(self.classes_):
            probas[:, j] = self.models[c].predict_proba(X)

        # normalización fila por fila (softmax-like)
        row_sums = probas.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return probas / row_sums

    def predict(self, X):
        probas = self.predict_proba(X)
        idx = np.argmax(probas, axis=1)
        return self.classes_[idx]