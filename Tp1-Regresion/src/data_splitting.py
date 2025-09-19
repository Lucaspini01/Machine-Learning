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