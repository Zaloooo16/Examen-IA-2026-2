"""
preprocessing.py
------------------
Preprocesamiento compartido: separa features/target, hace el split
Train/Test (80/20) y aplica StandardScaler ajustándolo ÚNICAMENTE
sobre el conjunto de entrenamiento (para evitar data leakage).
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def separar_features_target(df, target_col="target"):
    """Separa el DataFrame en X (features) e y (target)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def dividir_train_test(X, y, test_size=0.2, random_state=42):
    """Divide en Train/Test (80/20 por defecto)."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def escalar_datos(X_train, X_test):
    """
    Ajusta el StandardScaler ÚNICAMENTE con X_train y transforma
    tanto X_train como X_test con esos mismos parámetros.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform SOLO en train
    X_test_scaled = scaler.transform(X_test)          # transform (sin fit) en test
    return X_train_scaled, X_test_scaled, scaler
