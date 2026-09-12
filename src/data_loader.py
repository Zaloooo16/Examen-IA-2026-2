"""
data_loader.py
----------------
Módulo 1: Generación de Datos.

Genera un dataset sintético de clasificación con scikit-learn
(4 características continuas + 1 variable objetivo binaria),
lo guarda en data/dataset.csv y expone una función para cargarlo.
"""

import os
import pandas as pd
from sklearn.datasets import make_classification

# Rutas relativas a la raíz del proyecto
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")


def generar_dataset(n_samples=500, n_features=4, random_state=42):
    """
    Genera un dataset sintético de clasificación binaria con 4
    características continuas usando make_classification.

    Retorna un DataFrame de pandas con las columnas:
    feature_1, feature_2, feature_3, feature_4, target
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=random_state,
    )

    columnas = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columnas)
    df["target"] = y
    return df


def guardar_dataset(df: pd.DataFrame, path: str = DATASET_PATH):
    """Guarda el DataFrame en un archivo CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[data_loader] Dataset guardado en: {path}")


def cargar_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Carga el dataset desde el CSV. Si no existe, lo genera y lo
    guarda primero (para que el pipeline funcione de punta a punta
    con solo ejecutar main.py).
    """
    if not os.path.exists(path):
        print("[data_loader] No se encontró dataset.csv, generando uno nuevo...")
        df = generar_dataset()
        guardar_dataset(df, path)
        return df

    print(f"[data_loader] Cargando dataset desde: {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    # Permite ejecutar este archivo solo para generar el dataset
    df = generar_dataset()
    guardar_dataset(df)
    print(df.head())
