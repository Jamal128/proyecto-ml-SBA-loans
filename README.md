# Proyecto de Machine Learning: Predicción de Incumplimiento de Préstamos SBA
![image](https://github.com/user-attachments/assets/00422869-a209-45c1-8d2f-8014914cecfc)

## 📚 Descripción del Proyecto

Este proyecto individual, desarrollado como parte del módulo de Machine Learning del bootcamp de Data Science, tiene como objetivo principal la creación de un modelo predictivo para identificar la probabilidad de incumplimiento (`CHGOFF`) en préstamos otorgados por la U.S. Small Business Administration (SBA).

El modelo busca ayudar a las instituciones financieras a gestionar de manera más proactiva el riesgo crediticio, optimizar la asignación de capital y minimizar las pérdidas asociadas a los préstamos fallidos.

## 📊 Dataset

El análisis se basa en el dataset **"National SBA"**, un conjunto de datos histórico de préstamos garantizados por la SBA entre 1987 y 2014. Este dataset de tipo censal incluye:

* **Observaciones:** 899.164 préstamos individuales.
* **Variables:** 27 características detalladas sobre el préstamo, el prestatario y la institución bancaria.
* **Variable Objetivo:** `MIS_Status`, que indica si el préstamo fue 'Pagado en su totalidad' (`P I F`) o 'Incumplió' (`CHGOFF`).

**Nota Importante sobre los Datos de Entrenamiento:**
Debido al tamaño considerable del archivo original de entrenamiento, el conjunto de datos `train.csv` ha sido dividido en dos partes: `train_part1.csv` y `train_part2.csv`. Ambos archivos se encuentran en la carpeta `src/data/`. Para utilizar el conjunto de entrenamiento completo, deberás cargarlos y concatenarlos en tu notebook. El conjunto de prueba (`test.csv`) se encuentra en un único archivo.

## 🚀 Workflow de Machine Learning

El proyecto sigue un workflow completo de Machine Learning, incluyendo:

1.  **Recogida y Carga de Datos:** Adquisición del dataset "National SBA".
2.  **Limpieza de Datos:** Tratamiento de valores nulos, formatos inconsistentes y valores atípicos.
3.  **Análisis Exploratorio de Datos (EDA):** Visualización y entendimiento de las distribuciones de las variables, relaciones y el desequilibrio de la clase objetivo (`MIS_Status`).
4.  **Feature Engineering:** Creación de nuevas características a partir de las existentes y selección de las variables más relevantes para el modelo.
5.  **Preprocesamiento:** Manejo de variables categóricas (One-Hot Encoding) y numéricas (escalado). Abordaje del desequilibrio de clases.
6.  **Modelado:** Implementación de modelos de clasificación, con un enfoque principal en **XGBoost**.
7.  **Optimización de Hiperparámetros:** Uso de técnicas avanzadas (ej. Optuna) para ajustar los parámetros del modelo.
8.  **Evaluación e Interpretación del Modelo:** Análisis de métricas clave (AUC, Precision, Recall, F1-Score) y matriz de confusión. Interpretación de la importancia de las características.
9.  **Impacto en el Negocio:** Discusión de cómo el modelo puede aplicarse para mejorar la toma de decisiones financieras.

## 📦 Estructura del Repositorio

El código y los recursos del proyecto están organizados de la siguiente manera:
ProyectoML-SBA/
<img width="515" alt="image" src="https://github.com/user-attachments/assets/256d680e-ec2b-41d6-bdf4-d57b7246a5cd" />

## ⚠️ Nota Importante sobre el Modelo Principal

El modelo principal entrenado (`model_clasificacion.pkl`), que debería residir en `src/model/production/`, **no ha sido subido directamente a este repositorio de GitHub** debido a su tamaño considerable (que excede los límites permitidos para archivos individuales en GitHub sin configuraciones especiales de Git LFS avanzadas, o el límite de Git LFS gratuito para este tipo de proyectos).

Sin embargo, el código para su entrenamiento y el proceso para guardarlo están disponibles en los notebooks correspondientes dentro de `src/notebooks/`. Podrás replicar el entrenamiento y generar el archivo del modelo localmente.
