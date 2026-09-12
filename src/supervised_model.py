"""
supervised_model.py
----------------------
Módulo 2: Pipeline Supervisado.

Divide los datos en Train/Test (80/20), escala con StandardScaler
(ajustado solo en train), entrena un clasificador de Regresión
Logística y reporta Accuracy, F1-score y Matriz de Confusión.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.preprocessing import separar_features_target, dividir_train_test, escalar_datos


def entrenar_y_evaluar(df):
    """
    Ejecuta el pipeline supervisado completo sobre el DataFrame.

    Retorna un diccionario con el modelo entrenado y las métricas
    obtenidas, para poder reutilizarlas en el README/reporte.
    """
    # 1. Separar features y target
    X, y = separar_features_target(df)

    # 2. Split Train/Test 80/20
    X_train, X_test, y_train, y_test = dividir_train_test(X, y, test_size=0.2)

    # 3. Escalado (fit SOLO en train, transform en ambos)
    X_train_scaled, X_test_scaled, scaler = escalar_datos(X_train, X_test)

    # 4. Entrenamiento del clasificador
    modelo = LogisticRegression(random_state=42)
    modelo.fit(X_train_scaled, y_train)

    # 5. Predicción y métricas
    y_pred = modelo.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    matriz_confusion = confusion_matrix(y_test, y_pred)
    reporte = classification_report(y_test, y_pred)

    resultados = {
        "modelo": modelo,
        "scaler": scaler,
        "accuracy": accuracy,
        "f1_score": f1,
        "matriz_confusion": matriz_confusion,
        "reporte": reporte,
    }
    return resultados


def imprimir_resultados(resultados):
    """Imprime las métricas de forma legible (para copiar al README)."""
    print("\n===== PIPELINE SUPERVISADO (Regresión Logística) =====")
    print(f"Accuracy : {resultados['accuracy']:.4f}")
    print(f"F1-score : {resultados['f1_score']:.4f}")
    print("Matriz de Confusión:")
    print(resultados["matriz_confusion"])
    print("\nReporte de clasificación:")
    print(resultados["reporte"])
