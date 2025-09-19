import numpy as np
import pandas as pd

def one_hot_encoder(y: np.ndarray) -> np.ndarray:
    """
    Aplica one-hot encoding a un vector de etiquetas.
    """
    classes = np.unique(y)
    return np.eye(len(classes))[np.searchsorted(classes, y)]

def encode_tipo_column(df):
    """
    Transforma la columna 'tipo' en variables binarias (dummies), elimina la primera categoría y la columna original.
    """
    tipo_dummies = pd.get_dummies(df['tipo'], drop_first=True)
    df_encoded = pd.concat([df, tipo_dummies], axis=1)
    df_encoded.drop(columns=['tipo'], inplace=True)
    return df_encoded

def build_dataframe(X: np.ndarray, y: np.ndarray, feature_names: list, target_name: str = "precio") -> pd.DataFrame:
    """
    Convierte arrays de numpy X, y en un DataFrame de pandas con nombres de columnas.

    Parámetros
    ----------
    X : np.ndarray
        Matriz de features de tamaño (n_samples, n_features).
    y : np.ndarray
        Vector target de tamaño (n_samples,) o (n_samples, 1).
    feature_names : list
        Lista con los nombres de las columnas de X.
    target_name : str
        Nombre de la columna target (default = "precio").

    Retorna
    -------
    df : pd.DataFrame
        DataFrame con las features y el target.
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    if X.shape[1] != len(feature_names):
        raise ValueError(f"Dimensión inconsistente: X tiene {X.shape[1]} columnas pero se pasaron {len(feature_names)} nombres")

    df = pd.DataFrame(X, columns=feature_names)
    df[target_name] = y

    return df

import pandas as pd

def normalize(train_df: pd.DataFrame, val_df: pd.DataFrame, target_name: str = "precio"):
    """
    Estandariza columnas numéricas de train/val usando media y desvío de train.
    No toca la columna target.

    Parámetros
    ----------
    train_df : pd.DataFrame
        DataFrame de entrenamiento (features + target).
    val_df : pd.DataFrame
        DataFrame de validación (features + target).
    target_name : str
        Nombre de la columna target (default = "precio").

    Retorna
    -------
    train_norm : pd.DataFrame
        DataFrame normalizado (train).
    val_norm : pd.DataFrame
        DataFrame normalizado (val).
    mu : pd.Series
        Medias de train (solo features numéricas).
    sigma : pd.Series
        Desvíos estándar de train (solo features numéricas).
    """

    # Columnas numéricas sin el target
    num_cols = train_df.select_dtypes(include="number").columns.drop(target_name)

    # Calcular medias y desvíos en TRAIN
    mu = train_df[num_cols].mean()
    sigma = train_df[num_cols].std(ddof=0)
    sigma[sigma == 0.0] = 1.0  # evita división por 0

    # Normalizar
    train_norm = train_df.copy()
    val_norm = val_df.copy()

    train_norm[num_cols] = (train_norm[num_cols] - mu) / sigma
    val_norm[num_cols]   = (val_norm[num_cols] - mu) / sigma

    return train_norm, val_norm, mu, sigma
