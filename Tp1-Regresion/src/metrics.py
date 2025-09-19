import numpy as np
import src.models as models

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el error cuadrático medio (MSE) entre las predicciones y los valores reales.
    """
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el error absoluto medio (MAE) entre las predicciones y los valores reales.
    """
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula la raíz del error cuadrático medio (RMSE) entre las predicciones y los valores reales.
    """
    return np.sqrt(mse(y_true, y_pred))

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el coeficiente de determinación (R^2) entre las predicciones y los valores reales.
    """
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total) if ss_total > 0 else 0
