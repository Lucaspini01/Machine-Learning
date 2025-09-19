import numpy as np
import pandas as pd

def split_train_val(df, train_frac=0.8, random_state=42):
    """
    Mezcla y divide un DataFrame en conjuntos de entrenamiento y validación.

    Args:
        df (pd.DataFrame): DataFrame limpio a dividir.
        train_frac (float): Fracción de datos para entrenamiento.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        train_df (pd.DataFrame): Datos de entrenamiento.
        val_df (pd.DataFrame): Datos de validación.
    """
    df_shuffled = df.sample(frac=1, random_state=random_state)
    split_idx = int(train_frac * len(df_shuffled))
    train_df = df_shuffled.iloc[:split_idx].copy()
    val_df = df_shuffled.iloc[split_idx:].copy()
    return train_df, val_df

def add_missing_values_knn(df: pd.DataFrame, columns, k: int = 5):
    df_knn = df.copy()

    # Only use numeric columns to compute distances


    numeric_cols = columns
    for col in columns:
        
        features = df_knn[numeric_cols].drop(columns=col).columns

        not_nan_neighbors = df_knn[df_knn[col].notna()].dropna(subset=features)
        nan_neighbors = df_knn[df_knn[col].isna()].dropna(subset=features)

        for idx in nan_neighbors.index:
            target_row = df_knn.loc[idx, features].values.astype(float)
            known_rows = not_nan_neighbors[features].values.astype(float)

            # Compute Euclidean distances
            distances = np.linalg.norm(known_rows - target_row, axis=1)

            # Get k nearest neighbors' values for the target column
            nearest_indices = distances.argsort()[:k]
            neighbor_values = not_nan_neighbors.iloc[nearest_indices][col].values

            # Impute with mean of k neighbors
            imputed_value = np.mean(neighbor_values)
            df_knn.at[idx, col] = imputed_value


    return df_knn


def add_missing_values_mode(df: pd.DataFrame, columns):
    """
    Imputa valores faltantes con la moda (valor más frecuente) de cada columna.
    - Apta para variables categóricas y numéricas.
    - Si hay empate de modas, toma la primera según el orden de pandas.
    - Devuelve una copia del DataFrame.
    """
    df_out = df.copy()
    if isinstance(columns, str):
        columns = [columns]
        
    for col in columns:
        if col not in df_out.columns:
            raise KeyError(f"Columna no encontrada: {col}")
        # Serie de modas (puede haber varias)
        modes = df_out[col].mode(dropna=True)
        if len(modes) == 0:
            # Columna completamente vacía: no hay con qué imputar
            continue
        mode_val = modes.iloc[0]
        df_out[col] = df_out[col].fillna(mode_val)
    return df_out

def handle_missing_values(df, columns, method='mean'):
    """
    Handle missing values for columns of a dataframe.
    """
    
    df_missing = df.copy()
    for col in columns:

        if method == 'knn':
            df_missing = add_missing_values_knn(df_missing, [col])

        else:
            raise ValueError("Method not recognized. Use 'mean', 'mode', 'knn', or 'drop'.")
    
    return df_missing

def build_dataframe(X: np.ndarray, y: np.ndarray, feature_names: list, target_name: str = "Diagnosis") -> pd.DataFrame:
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
        Nombre de la columna target (default = "Diagnosis").

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

def normalize(train_df: pd.DataFrame, val_df: pd.DataFrame, target_name: str = "Diagnosis"):
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
        Nombre de la columna target (default = "Diagnosis").

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