import pandas as pd
import numpy as np

def confusion_matrix(y_true, y_pred, labels):
    """
    Compute the confusion matrix.
    """
    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    label_to_index = {label: index for index, label in enumerate(labels)}

    for true, pred in zip(y_true, y_pred):
        if true in label_to_index and pred in label_to_index:
            cm[label_to_index[true], label_to_index[pred]] += 1

    return cm

def accuracy(y_true, y_pred):
    """
    Compute the accuracy score.
    """
    return np.sum(y_true == y_pred) / len(y_true)

def precision(y_true, y_pred, labels):
    """
    Compute the precision score for each class.
    """
    cm = confusion_matrix(y_true, y_pred, labels)
    precisions = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        precisions[label] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precisions

def recall(y_true, y_pred, labels):
    """
    Compute the recall score for each class.
    """
    cm = confusion_matrix(y_true, y_pred, labels)
    recalls = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        recalls[label] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recalls

def f1_score(y_true, y_pred, labels):
    """
    Compute the F1 score for each class.
    """
    precisions = precision(y_true, y_pred, labels)
    recalls = recall(y_true, y_pred, labels)
    f1_scores = {}
    for label in labels:
        p = precisions[label]
        r = recalls[label]
        f1_scores[label] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    return f1_scores



def auc_trapezoid(x, y):
    """
    Área bajo curva por regla trapezoidal.
    Requiere x ordenado de forma ascendente.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # ordenar por x por si acaso
    order = np.argsort(x)
    x, y = x[order], y[order]
    dx = np.diff(x)
    y_mid = (y[1:] + y[:-1]) / 2.0

    return float(np.sum(dx * y_mid))

def _binarize(y_true, positive_label):
    """Devuelve vector binario 0/1 para la clase positiva."""
    y_true = np.asarray(y_true)
    return (y_true == positive_label).astype(int)

def pr_curve_ovr(y_true, y_scores, positive_label):
    """
    Curva Precision-Recall para una clase (OvR) sin sklearn.
    y_scores: probas o puntajes (más alto = más positivo).
    Devuelve (precisions, recalls).
    """
    y_true_bin = _binarize(y_true, positive_label)
    scores = np.asarray(y_scores, dtype=float)

    # Ordenar por score descendente
    order = np.argsort(-scores)
    y_true_sorted = y_true_bin[order]
    scores_sorted = scores[order]

    # Barrido de umbral en puntos únicos de score
    # (con acumulados para evitar O(n^2))
    tp_cum = np.cumsum(y_true_sorted == 1)
    fp_cum = np.cumsum(y_true_sorted == 0)

    # Para cada posición i, si cortamos hasta i -> TP=tp_cum[i], FP=fp_cum[i]
    # FN = P - TP
    P = tp_cum[-1]  # cantidad de positivos verdaderos
    # Si no hay positivos, devolvemos curva degenerada
    if P == 0:
        return np.array([1.0]), np.array([0.0])

    # Tomar índices donde el score cambia (umbrales únicos)
    # Incluimos el último índice para cerrar la curva
    distinct_mask = np.r_[True, scores_sorted[1:] != scores_sorted[:-1]]
    idxs = np.where(distinct_mask)[0]

    precisions = []
    recalls = []
    for i in idxs:
        TP = tp_cum[i]
        FP = fp_cum[i]
        FN = P - TP
        prec = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

    # Por convención, agregamos el punto (recall=0, precision=pos_rate) para cerrar
    # y (recall=1, precision=TP/(TP+FP) al threshold mínimo ya está incluido.
    precisions = np.array(precisions, dtype=float)
    recalls = np.array(recalls, dtype=float)

    # Asegurar orden creciente de recall para AUC
    order_r = np.argsort(recalls)
    return precisions[order_r], recalls[order_r]

def roc_curve_ovr(y_true, y_scores, positive_label):
    """
    Curva ROC para una clase (OvR) sin sklearn.
    Devuelve (fpr, tpr).
    """
    y_true_bin = _binarize(y_true, positive_label)
    scores = np.asarray(y_scores, dtype=float)

    order = np.argsort(-scores)
    y_true_sorted = y_true_bin[order]
    scores_sorted = scores[order]

    tp_cum = np.cumsum(y_true_sorted == 1)
    fp_cum = np.cumsum(y_true_sorted == 0)

    P = tp_cum[-1]
    N = fp_cum[-1]
    if P == 0 or N == 0:
        # Curva degenerada si falta una clase
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    distinct_mask = np.r_[True, scores_sorted[1:] != scores_sorted[:-1]]
    idxs = np.where(distinct_mask)[0]

    tpr_list = []
    fpr_list = []
    for i in idxs:
        TP = tp_cum[i]
        FP = fp_cum[i]
        TPR = TP / P
        FPR = FP / N
        tpr_list.append(TPR)
        fpr_list.append(FPR)

    # Añadir (0,0) y (1,1) para completar la curva
    fpr = np.array([0.0] + fpr_list + [1.0], dtype=float)
    tpr = np.array([0.0] + tpr_list + [1.0], dtype=float)

    # Asegurar orden por FPR
    order_f = np.argsort(fpr)
    return fpr[order_f], tpr[order_f]

def auc_pr_ovr(y_true, y_scores, positive_label):
    precisions, recalls = pr_curve_ovr(y_true, y_scores, positive_label)
    # AUC-PR integra precision vs recall (eje x = recall)
    return auc_trapezoid(recalls, precisions)

def auc_roc_ovr(y_true, y_scores, positive_label):
    fpr, tpr = roc_curve_ovr(y_true, y_scores, positive_label)
    # AUC-ROC integra tpr vs fpr (eje x = fpr)
    return auc_trapezoid(fpr, tpr)

