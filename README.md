# Aprendizaje de Máquina II

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

TP

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         tp_amq2_17co2024 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── tp_amq2_17co2024   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes tp_amq2_17co2024 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

# Criterios de aprobación

## Objetivos de la materia:

El objetivo está centrado en disponibilizar las herramientas de machine learning en un entorno productivo, utilizando herramientas de MLOPS.

## Evaluación

La evaluación de los conocimientos impartidos durante las clases será a modo de entrega de un trabajo práctico final. El trabajo es grupal (máximo 6 personas, mínimo 2 personas).

La idea de este trabajo es suponer que trabajamos para **ML Models and something more Inc.**, la cual ofrece un servicio que proporciona modelos mediante una REST API. Internamente, tanto para realizar tareas de DataOps como de MLOps, la empresa cuenta con Apache Airflow y MLflow. También dispone de un Data Lake en S3.

Ofrecemos tres tipos de evaluaciones:

 * **Nivel fácil** (nota entre 4 y 5): Hacer funcionar el ejemplo de aplicación [example_implementation](https://github.com/facundolucianna/amq2-service-ml/tree/example_implementation). Filmar un video del sistema funcionando:
   * Ejecutar en Airflow el DAG llamado `process_etl_heart_data`
   * Ejecuta la notebook (ubicada en `notebook_example`) para realizar la búsqueda de hiperparámetros y entrenar el mejor modelo.
   * Utilizar la REST-API del modelo.
   * Ejecutar en Airflow el DAG llamado `process_etl_heart_data` y luego `retrain_the_model`.
   * En todos estos pasos verificar lo que muestra MLFlow.
 * **Nivel medio** (nota entre 6 y 8): Implementar en local usando Metaflow el ciclo de desarrollo del modelo que desarrollaron en Aprendizaje de Máquina I y generar un archivo para predicción en bache (un csv o un archivo de SQLite). Sería implementar algo parecido a [batch_example](https://github.com/facundolucianna/amq2-service-ml/tree/batch_example), pero sin la parte del servicio de Docker. La nota puede llegar a 10 si implementan una base de datos (ya sea KVS u otro tipo) con los datos de la predicción en bache.
 * **Nivel alto** (nota entre 8 y 10): Implementar el modelo que desarrollaron en Aprendizaje de Máquina I en este ambiente productivo. Para ello, pueden usar los recursos que consideren apropiado. Los servicios disponibles de base son Apache Airflow, MLflow, PostgresSQL, MinIO, FastAPI. Todo está montado en Docker, por lo que además deben instalado Docker. 

### Repositorio con el material

Las herramientas para poder armar el proyecto se encuentra en: 
[https://github.com/facundolucianna/amq2-service-ml](https://github.com/facundolucianna/amq2-service-ml).

Además, dejamos un ejemplo de aplicación en el branch [example_implementation](https://github.com/facundolucianna/amq2-service-ml/tree/example_implementation).

## Criterios de aprobación

Los criterios de aprobación son los siguientes:

1. La entrega consiste en un repositorio en Github o Gitlab con la implementación y documentación. Salvo el nivel fácil que es un link al video.
2. La fecha de entrega máxima es 7 días después de la última clase.
3. El trabajo es obligatorio ser grupal para evaluar la dinámica de trabajo en un equipo de trabajo tipico.
4. La implementación debe de estar de acuerdo al nivel elegido. Sí es importante además de la implementación, hacer una buena documentación.
5. Son libres de incorporar o cambiar de tecnologías, pero es importante que lo implementado tenga un servicio de orquestación y algún servicio de ciclo de vida de modelos.   
6. La entrega es por medio del aula virtual de la asignatura y solo debe enviarse el link al repositorio.

