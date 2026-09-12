"""
main.py
---------
Orquestador principal del pipeline.

Ejecuta en orden:
1. Generación/carga del dataset (src/data_loader.py)
2. Pipeline supervisado (src/supervised_model.py)
3. Pipeline no supervisado (src/unsupervised_model.py)

Ejecutar con:  python main.py
"""

from src.data_loader import cargar_dataset
from src.supervised_model import entrenar_y_evaluar, imprimir_resultados as imprimir_supervisado
from src.unsupervised_model import ejecutar_pipeline_no_supervisado, imprimir_resultados as imprimir_no_supervisado


def main():
    print("========================================")
    print(" PIPELINE DE MACHINE LEARNING - EXAMEN IA")
    print("========================================")

    # 1. Generación / carga de datos
    df = cargar_dataset()
    print(f"\nDataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(df.head())

    # 2. Pipeline Supervisado
    resultados_sup = entrenar_y_evaluar(df)
    imprimir_supervisado(resultados_sup)

    # 3. Pipeline No Supervisado
    resultados_no_sup = ejecutar_pipeline_no_supervisado(df, k=3)
    imprimir_no_supervisado(resultados_no_sup)

    print("\n========================================")
    print(" PIPELINE FINALIZADO CORRECTAMENTE")
    print("========================================")


if __name__ == "__main__":
    main()
