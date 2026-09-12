"""
unsupervised_model.py
-------------------------
Módulo 3: Pipeline No Supervisado.

Aplica PCA (4 -> 2 componentes), K-Means (k=3) sobre las 2
componentes principales, calcula el Coeficiente de Silueta y la
varianza explicada acumulada, y genera una visualización 2D del
agrupamiento.
"""

import os
import matplotlib
matplotlib.use("Agg")  # backend sin GUI, para poder guardar la imagen a archivo
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.preprocessing import separar_features_target

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PLOT_PATH = os.path.join(OUTPUT_DIR, "clustering_pca.png")


def aplicar_pca(X, n_components=2):
    """Reduce las features a n_components (2) con PCA."""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    varianza_explicada = pca.explained_variance_ratio_
    varianza_acumulada = varianza_explicada.sum()
    return X_pca, pca, varianza_acumulada


def aplicar_kmeans(X_pca, k=3):
    """Aplica K-Means sobre las componentes principales."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    return kmeans, labels


def graficar_clusters(X_pca, labels, path=PLOT_PATH):
    """Genera y guarda una visualización 2D de los clusters."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=40)
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.title("Clustering K-Means (k=3) sobre componentes PCA")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[unsupervised_model] Gráfico guardado en: {path}")


def ejecutar_pipeline_no_supervisado(df, k=3):
    """Ejecuta PCA + K-Means + Silueta sobre el DataFrame completo."""
    X, _ = separar_features_target(df)  # no se usa el target (no supervisado)

    X_pca, pca, varianza_acumulada = aplicar_pca(X, n_components=2)
    kmeans, labels = aplicar_kmeans(X_pca, k=k)
    silueta = silhouette_score(X_pca, labels)

    graficar_clusters(X_pca, labels)

    resultados = {
        "varianza_acumulada": varianza_acumulada,
        "silhouette_score": silueta,
        "labels": labels,
    }
    return resultados


def imprimir_resultados(resultados):
    """Imprime las métricas del pipeline no supervisado."""
    print("\n===== PIPELINE NO SUPERVISADO (PCA + K-Means) =====")
    print(f"Varianza explicada acumulada (2 componentes): {resultados['varianza_acumulada']:.4f}")
    print(f"Coeficiente de Silueta (Silhouette Score): {resultados['silhouette_score']:.4f}")
