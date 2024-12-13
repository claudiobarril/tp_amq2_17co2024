# Predictor del precio de automóviles - **ML Models and something more Inc.**

## Descripción del Proyecto

Este proyecto tiene como objetivo ayudar a los usuarios a predecir el precio de venta de su automóvil usado, a través de una interfaz amigable y utilizando la potencia de la inteligencia artificial.

## Requisitos Previos

- Docker y Docker Compose instalados.
- Variables de entorno configuradas.

## Configuración Inicial

### Variables de Entorno

Cree un archivo `.env` en la raíz del proyecto con las siguientes variables:

```env
# airflow configuration
AIRFLOW_UID=50000
AIRFLOW_GID=0
AIRFLOW_PROJ_DIR=./airflow
AIRFLOW_PORT=8080
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow

# postgres configuration
PG_USER=airflow
PG_PASSWORD=airflow
PG_DATABASE=airflow
PG_PORT=5432

# mlflow configuration
MLFLOW_PORT=5002
MLFLOW_S3_ENDPOINT_URL=http://s3:9000

# minio configuration
MINIO_ACCESS_KEY=minio
MINIO_SECRET_ACCESS_KEY=minio123
MINIO_PORT=9000
MINIO_PORT_UI=9001
MLFLOW_BUCKET_NAME=mlflow
DATA_REPO_BUCKET_NAME=data

# fastapi configuration
FASTAPI_PORT=8800

# fastapi configuration
REDIS_HOST=redis
REDIS_PORT=6379
```

## Servicios Incluidos

- **Airflow**: [http://localhost:8080](http://localhost:8080) 
  - DAGs disponibles:
    - `el_process`: Extracción y carga de datos desde Kaggle a S3.
    - `lt_process`: Procesamiento de datos y actualización de pipelines.
    - `train_the_model`: Entrenamiento del modelo XGBoost.
    - `train_the_catboost_model`: Entrenamiento del modelo CatBoost.
    - `retrain_the_model`: Reentrenamiento del modelo XGBoost.
    - `retrain_the_catboost_model`: Reentrenamiento del modelo CatBoost.
    - `daily_batch_processing`: Procesamiento batch diario de solicitudes.
    - `batch_processing`: Procesamiento completo de datos históricos.
    - `batch_processing_catboost`: Procesamiento completo con CatBoost.
- **MLflow**: [http://localhost:5002](http://localhost:5002)
  - Experimento: `Cars`
  - Modelos:
    - `xgboost_dev` y `xgboost_prod`
    - `catboost_dev` y `catboost_prod`
- **S3 (Minio)**: [http://localhost:9001](http://localhost:9001)
  - Estructura:
    - `data/raw`: Dataset crudo extraído de Kaggle.
    - `data/final`: Datos procesados y predicciones realizadas.
    - `pipeline`: Pipelines entrenados para transformación de datos.
- **FastAPI Backend**: [http://localhost:8800](http://localhost:8800)
  - Endpoints disponibles:
    - `GET /`: Página de inicio del proyecto.
    - `POST /predict`: Predicción a partir de datos enviados por los usuarios.
- **Frontend**: [http://localhost:8800](http://localhost:8800)
  - Formulario amigable para la carga de datos de predicción.

## Pasos de Ejecución

### Primera vez

Dado que la aplicación predictora necesita de un pipeline entrenado para pre-procesar los datos de entrada del usuario,
y un modelo predictivo, lo ideal es levantar primero airflow y ejecutar los procesos que entrenan el mejor modelo con el
que se cuenta actualmente.

### 1. Levantar el Perfil `airflow`

Ejecute el siguiente comando para iniciar solo los servicios necesarios para Airflow:

```bash
docker-compose -f docker-compose.yml --profile airflow up --build
```

### 2. Ejecutar los DAGs necesarios

Acceda a la interfaz web de Airflow en [http://localhost:8080](http://localhost:8080). Asegúrese de que los DAGs estén habilitados y ejecútelos en el siguiente orden:

1. **process_el_cars_data**: Extrae el dataset actualizado desde Kaggle y lo carga en el bucket S3.
2. **process_lt_cars_data**: Carga desde S3 los datos extraídos y las predicciones realizadas, los procesa y guarda el pipeline entrenado en S3 para su uso posterior.
3. **train_the_model**: Entrena un modelo XGBoost, que ha sido seleccionado como principal tras iteraciones previas del equipo.
4. **batch_processing_model**: Guarda en Redis todos los datos de Kaggle y las predicciones realizadas hasta el momento. Este paso no es necesario, pero si recomendable para contar ya con una buena base de predicciones pre-calculadas.


### 3. Verificar Resultados

- **Pipeline**: Verifique que el pipeline esté disponible en su bucket S3 en la carpeta correspondiente.
- **Modelo**: Confirme que el modelo `cars_model_prod` esté registrado en MLflow.

### 4. Levantar el Perfil `predictor`

Una vez completados los pasos anteriores, puede levantar el perfil `predictor`:

```bash
docker-compose -f docker-compose.yml --profile predictor up --build
```

### Como levantar todos los servicios

Si ya cuenta con un pipeline y un modelo disponibles, puede simplemente levantar todos los servicios utilizando el perfil `all`:

```bash
docker-compose -f docker-compose.yml --profile predictor up --build
```

Esto incluirá los servicios de FastAPI y el frontend para realizar predicciones.

## Comandos Útiles

- Detener todos los servicios:
  ```bash
  docker-compose down
  ```
- Ver logs de un servicio específico:
  ```bash
  docker-compose logs <nombre_servicio>
  ```

## Estructura del Proyecto - (REVISAR)

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

## Notas

- Asegúrese de contar con los permisos necesarios para acceder al bucket S3.
- Revise que los DAGs estén correctamente configurados antes de su ejecución.
- Verifique que los datos y modelos estén correctamente registrados en las ubicaciones esperadas.

## Contacto

Para preguntas o soporte, contacte al equipo de desarrollo.
