# Proyecto Final: Predicción y Generación de Insights sobre Calidad de Vinos

## MLOps con MLflow + Gen AI + Gradio

Este proyecto implementa un pipeline completo de MLOps para predecir la calidad de vinos blancos a partir de sus propiedades químicas, integrando experimentación reproducible, registro de modelos, interfaz de usuario interactiva y explicaciones generadas por inteligencia artificial.

---

## Objetivo

Desarrollar un sistema end-to-end que:

- Entrene y compare modelos de machine learning con MLflow Tracking.
- Registre y versione el mejor modelo en MLflow Model Registry.
- Exponga predicciones a usuarios finales mediante una interfaz Gradio.
- Genere explicaciones automáticas usando un LLM local (Ollama) para mejorar la interpretabilidad.
- Siga buenas prácticas de reproducibilidad, modularidad y documentación.

---

## Tecnologías Utilizadas

- **MLflow**: Tracking, Projects, Model Registry, Artifacts
- **Scikit-learn**: Random Forest Regressor
- **Gradio**: Interfaz web interactiva
- **Ollama (con llama3)**: Generación de explicaciones con IA generativa
- **Pandas / NumPy / Matplotlib**: Análisis y visualización
- **Conda**: Gestión de entorno reproducible

---

## Estructura del Proyecto

```text
├── data/ # Dataset: winequality-white.csv
├── notebooks/ # Análisis exploratorio (Jupyter)
├── src/ # Código fuente modular
│ ├── train.py # Entrenamiento con MLflow Tracking
│ ├── utils.py # Funciones compartidas (carga de datos, etc.)
│ ├── genai_explainer.py # Módulo de explicaciones con LLM
│ └── evaluate.py # (Opcional) Evaluación post-registro
├── gradio_app/ # Aplicación de usuario final
│ └── app.py # Interfaz con modelo desde Model Registry
├── MLproject # Definición de MLflow Project
├── conda.yaml # Entorno reproducible
└── README.md # Este archivo
```

---

## Cómo Ejecutar el Proyecto

### 1. Clonar y configurar el entorno

```bash
git clone <tu-repositorio>
cd <carpeta-del-proyecto>
conda env create -f conda.yaml
conda activate wine-mlflow-genai
```

Asegúrate de tener Ollama instalado y el modelo llama3 descargado:

```bash
ollama pull llama3
```

### 2. Entrenar modelos (3 experimentos con hiperparámetros distintos)

```bash
mlflow run . -P n_estimators=50  -P max_depth=5
mlflow run . -P n_estimators=100 -P max_depth=10
mlflow run . -P n_estimators=200 -P max_depth=None
```

### 3. Visualizar experimentos y promover a producción

```bash
mlflow ui
```

Luego accede a: [http://localhost:5000](http://localhost:5000)

- Compara los runs
- Registra el mejor modelo como WineQualityModel
- Promueve una versión a Staging, luego a Production

### 4. Ejecutar la interfaz de usuario

```bash
cd gradio_app
python app.py
```

Tu navegador abrirá: [http://localhost:7860](http://localhost:7860)

- Ingresa valores o usa ejemplos predefinidos
- ¡Obtén predicciones + explicaciones generadas por IA!

---

## Dataset

**Fuente**: UCI Machine Learning Repository – Wine Quality

- **Características**: 11 atributos químicos (acidez, alcohol, azúcar residual, etc.)
- **Objetivo**: Calidad del vino (escala entera 0–10)
- **Tipo de problema**: Regresión

---

## Generative AI (Explicaciones)

Cuando se realiza una predicción, el sistema llama a Ollama + Llama3 para generar una explicación como:

> "Este vino tiene alta calidad porque presenta un contenido de alcohol elevado (10.5%) y baja acidez volátil (0.27 g/dm³), lo que indica equilibrio y estabilidad."

Además, la explicación se registra automáticamente como artefacto en MLflow.

---

## Entrega Final

Este repositorio incluye:

- ✅ Código fuente modular y reproducible
- ✅ Notebook de exploración de datos
- ✅ Interfaz Gradio funcional
- ✅ Capturas de MLflow UI (incluidas en el informe)

---

## Autores

**Nombres**: Juan Mosquera, Anderson Bornachera  
**Curso**: MLOps – Universidad del Magdalena  
**Fecha**: Noviembre 2025
