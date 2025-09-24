import numpy as np
import pandas as pd

def split_train_test(df, train_frac=0.8, random_state=42):
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
    test_df = df_shuffled.iloc[split_idx:].copy()
    return train_df, test_df

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

def normalize_test(test_df: pd.DataFrame, given_means: pd.Series, given_stds: pd.Series, target_name: str = "Diagnosis"):  
    """
    Estandariza columnas numéricas de test usando medias y desvíos dados (de train).
    No toca la columna target.

    Parámetros
    ----------
    test_df : pd.DataFrame
        DataFrame de test (features + target).
    given_means : pd.Series
        Medias pre-calculadas (solo features numéricas).
    given_stds : pd.Series
        Desvíos estándar pre-calculados (solo features numéricas).
    target_name : str
        Nombre de la columna target (default = "Diagnosis").

    Retorna
    -------
    test_norm : pd.DataFrame
        DataFrame normalizado (test).
    """
    
    # Columnas numéricas sin el target
    num_cols = test_df.select_dtypes(include="number").columns.drop(target_name)

    # Normalizar
    test_norm = test_df.copy()
    
    for column in num_cols:
        if column in given_means.index and column in given_stds.index:
            if given_stds[column] == 0.0:
                test_norm[column] = test_norm[column] - given_means[column]
            else:
                test_norm[column] = (test_norm[column] - given_means[column]) / given_stds[column]
        else:
            raise KeyError(f"Column '{column}' not found in given means or stds.")

    return test_norm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, chi2_contingency

# ========== Rangos de variables ==========
def feature_ranges(df, exclude_cols=None):
    """
    Devuelve un DataFrame con min, max, media, std para cada feature numérica.
    """
    if exclude_cols is None:
        exclude_cols = []
    num_df = df.drop(columns=exclude_cols, errors="ignore").select_dtypes(include=[np.number])
    summary = num_df.describe().T[["min", "max", "mean", "std"]]
    return summary

# ========== Distribuciones vs target ==========
def plot_feature_distributions(df, target, numeric_cols=None, categorical_cols=None):
    """
    Plotea histogramas/boxplots para numéricas y barras para categóricas, divididos por target.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore").columns
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=target, y=col, data=df)
        plt.title(f"{col} vs {target}")
        plt.show()

    for col in categorical_cols:
        plt.figure(figsize=(6,4))
        sns.countplot(x=col, hue=target, data=df)
        plt.title(f"{col} vs {target}")
        plt.show()

# ========== Correlaciones ==========
def correlation_with_target(df, target, numeric_cols=None, categorical_cols=None):
    """
    Calcula correlación de cada feature con el target.
    - Numéricas: correlación punto-biserial.
    - Categóricas: chi-cuadrado (p-valor).
    Maneja valores NaN ignorando filas con NaN en la feature o el target.
    """
    results = {}
    y_full = df[target].values

    # Numéricas
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore").columns
    for col in numeric_cols:
        # Elimina filas con NaN en la feature o el target
        mask = df[col].notna() & df[target].notna()
        if mask.sum() == 0:
            results[col] = {"type": "numeric", "correlation": np.nan, "pval": np.nan}
            continue
        y = df.loc[mask, target].values
        x = df.loc[mask, col].values
        corr, pval = pointbiserialr(y, x)
        results[col] = {"type": "numeric", "correlation": corr, "pval": pval}

    # Categóricas
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        # Elimina filas con NaN en la feature o el target
        mask = df[col].notna() & df[target].notna()
        if mask.sum() == 0:
            results[col] = {"type": "categorical", "chi2": np.nan, "pval": np.nan}
            continue
        contingency = pd.crosstab(df.loc[mask, col], df.loc[mask, target])
        if contingency.empty:
            results[col] = {"type": "categorical", "chi2": np.nan, "pval": np.nan}
            continue
        chi2, pval, _, _ = chi2_contingency(contingency)
        results[col] = {"type": "categorical", "chi2": chi2, "pval": pval}

    return pd.DataFrame(results).T

import numpy as np
import pandas as pd

# DETECCIÓN DE OUTLIERS

def compute_iqr_bounds(series, k=1.5):
    """ Devuelve (lb, ub) usando el método IQR: [Q1 - k*IQR, Q3 + k*IQR]. """
    q1 = np.nanpercentile(series, 25)
    q3 = np.nanpercentile(series, 75)
    iqr = q3 - q1
    lb = q1 - k * iqr
    ub = q3 + k * iqr
    return float(lb), float(ub)

def compute_zscore_bounds(series, z=3.0):
    """ Devuelve (lb, ub) usando media±z*std (no robusto a outliers fuertes). """
    mu = np.nanmean(series)
    sd = np.nanstd(series, ddof=0)
    if sd == 0 or np.isnan(sd):
        return float(mu), float(mu)  # sin varianza: no hay rango útil
    return float(mu - z * sd), float(mu + z * sd)

def compute_percentile_bounds(series, lo=1.0, hi=99.0):
    """ Devuelve (lb, ub) por percentiles. """
    lb = np.nanpercentile(series, lo)
    ub = np.nanpercentile(series, hi)
    return float(lb), float(ub)

def merge_bounds(primary, secondary):
    """
    Intersección conservadora de dos rangos (lb, ub).
    Si alguno es None, usa el otro.
    """
    if primary is None: return secondary
    if secondary is None: return primary
    lb = max(primary[0], secondary[0])
    ub = min(primary[1], secondary[1])
    if lb > ub:  # si no hay intersección, quedate con el primario
        return primary
    return (lb, ub)

# MANEJO DE OUTLIERS

def build_outlier_bounds(
    df,
    numeric_cols,
    method="iqr",
    iqr_k=1.5,
    z=3.0,
    p_lo=1.0,
    p_hi=99.0,
    theoretical_bounds=None,
    use_ranges_df=None
):
    """
    Calcula límites inferiores/superiores por columna para detectar/recortar outliers.

    Params:
    - numeric_cols: lista de columnas numéricas a evaluar.
    - method: 'iqr' | 'zscore' | 'percentile'
    - theoretical_bounds: dict opcional {col: (lb_teo, ub_teo)} con límites teóricos conocidos.
    - use_ranges_df: DataFrame opcional con min/max (p.ej. el devuelto por feature_ranges()).
                     Si se pasa, intersecta con (min, max) observados.

    Devuelve:
    - bounds: dict {col: (lb, ub)}
    """
    bounds = {}
    for col in numeric_cols:
        s = df[col].astype(float)

        if method == "iqr":
            lb, ub = compute_iqr_bounds(s, k=iqr_k)
        elif method == "zscore":
            lb, ub = compute_zscore_bounds(s, z=z)
        elif method == "percentile":
            lb, ub = compute_percentile_bounds(s, lo=p_lo, hi=p_hi)
        else:
            raise ValueError("method debe ser 'iqr', 'zscore' o 'percentile'.")

        
        if use_ranges_df is not None and col in use_ranges_df.index:
            obs_lb = float(use_ranges_df.loc[col, "min"])
            obs_ub = float(use_ranges_df.loc[col, "max"])
            lb, ub = merge_bounds((lb, ub), (obs_lb, obs_ub))

    
        if theoretical_bounds and col in theoretical_bounds:
            lb, ub = merge_bounds((lb, ub), theoretical_bounds[col])

        bounds[col] = (lb, ub)
    return bounds

def handle_outliers(
    df,
    bounds,
    action="clip"
):
    """
    Aplica el manejo de outliers según 'bounds' por columna.
    - action:
        'clip'       -> recorta valores fuera de [lb, ub] (winsorize duro)
        'remove'     -> elimina filas que violan al menos una regla
        'nan'        -> setea a NaN los out-of-range (para luego imputar)
    Devuelve (df_out, report) donde report resume cuántos valores afectó por columna.
    """
    out = df.copy()
    report = {}

    for col, (lb, ub) in bounds.items():
        s = out[col].astype(float)
        mask_low  = s < lb
        mask_high = s > ub
        n_low = int(mask_low.sum())
        n_high = int(mask_high.sum())

        if action == "clip":
            out.loc[mask_low, col]  = lb
            out.loc[mask_high, col] = ub
        elif action == "remove":
            # Lo hago despues
            pass
        elif action == "nan":
            out.loc[mask_low | mask_high, col] = np.nan
        else:
            raise ValueError("action debe ser 'clip', 'remove' o 'nan'.")

        report[col] = {"lb": lb, "ub": ub, "n_low": n_low, "n_high": n_high, "n_total": int(n_low + n_high)}

    if action == "remove":
        # elimino cualquier fila que haya violado algún bound
        to_remove = pd.Series(False, index=out.index)
        for col, (lb, ub) in bounds.items():
            s = out[col].astype(float)
            to_remove |= (s < lb) | (s > ub)
        n_rows_before = out.shape[0]
        out = out.loc[~to_remove].copy()
        report["_rows_removed"] = int(n_rows_before - out.shape[0])

    return out, pd.DataFrame(report).T


def suggest_numeric_columns(df, exclude=None):
    """ Devuelve columnas numéricas candidatas a outliers (excluyendo target u otras). """
    exclude = set(exclude or [])
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

def infer_cols(df, target):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols: num_cols.remove(target)
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols