import numpy as np
import matplotlib.pyplot as plt

# --- Accuracy ---
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# --- Matriz de confusión ---
def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# --- F1-Score Macro ---
def f1_score_macro(y_true, y_pred, num_classes):
    f1_per_class = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fp == 0 or tp + fn == 0:
            f1_per_class.append(0.0)
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            f1_per_class.append(f1)
    return np.mean(f1_per_class)



def metrics_report(X_train, Y_train_oh, y_train,
                   X_val, Y_val_oh, y_val,
                   parametros, num_classes):
    """
    Evalúa un modelo entrenado y muestra métricas de performance.
    Calcula Accuracy, Cross-Entropy, Matriz de Confusión y F1-Score Macro
    tanto para entrenamiento como validación.

    Parámetros
    ----------
    X_train, X_val : np.ndarray
        Entradas (flattened) para train y validation (shape: [n_features, n_samples])
    Y_train_oh, Y_val_oh : np.ndarray
        Etiquetas one-hot codificadas
    y_train, y_val : np.ndarray
        Etiquetas en formato entero
    parametros : dict
        Pesos del modelo entrenado
    num_classes : int
        Número total de clases
    """
    # === Predicciones ===
    y_train_pred = models.predict(X_train, parametros)
    y_val_pred   = models.predict(X_val, parametros)

    # === Accuracy ===
    acc_train = accuracy(y_train, y_train_pred)
    acc_val   = accuracy(y_val, y_val_pred)

    # === Cross-Entropy ===
    Y_hat_train, _ = models.forward_propagation(X_train, parametros)
    Y_hat_val, _   = models.forward_propagation(X_val, parametros)
    loss_train = models.cross_entropy_loss(Y_train_oh, Y_hat_train)
    loss_val   = models.cross_entropy_loss(Y_val_oh, Y_hat_val)

    # === Matriz de confusión y F1 ===
    cm_train = confusion_matrix(y_train, y_train_pred, num_classes)
    cm_val   = confusion_matrix(y_val, y_val_pred, num_classes)

    f1_train = f1_score_macro(y_train, y_train_pred, num_classes)
    f1_val   = f1_score_macro(y_val, y_val_pred, num_classes)

    # === Mostrar resultados ===
    print("Entrenamiento:")
    print(f"  Accuracy:       {acc_train:.4f}")
    print(f"  Cross-Entropy:  {loss_train:.4f}")
    print(f"  F1-Score Macro: {f1_train:.4f}")
    print("\nValidación:")
    print(f"  Accuracy:       {acc_val:.4f}")
    print(f"  Cross-Entropy:  {loss_val:.4f}")
    print(f"  F1-Score Macro: {f1_val:.4f}")

    return {
        "acc_train": acc_train, "acc_val": acc_val,
        "loss_train": loss_train, "loss_val": loss_val,
        "f1_train": f1_train, "f1_val": f1_val,
        "cm_train": cm_train, "cm_val": cm_val
    }
