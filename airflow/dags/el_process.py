import awswrangler as wr
import mlcroissant as mlc
import pandas as pd

from airflow.decorators import dag, task
from config.default_args import default_args  # Importa el archivo

markdown_text = """
### EL Process for Cars Data
"""


@dag(
    dag_id="process_el_cars_data",
    description="EL process for cars data, getting dataset from external source and storing it in s3.",
    doc_md=markdown_text,
    tags=["EL", "Cars"],
    default_args=default_args,
    catchup=False,
)
def process_el_cars_data():
    @task
    def get_data():
        """
        Load the raw data from the cars dataset.
        """
        import logging
        logger = logging.getLogger("airflow.task")
        try:
            # Cargar el dataset desde Kaggle usando Croissant
            croissant_dataset = mlc.Dataset(
                'http://www.kaggle.com/datasets/sajaabdalaal/car-details-v3csv/croissant/download')

            # Examinar los record sets disponibles
            record_sets = croissant_dataset.metadata.record_sets
            print(f"Conjuntos de registros disponibles: {record_sets}")
            logger.info("Conjuntos de registros disponibles: %s", record_sets)

            # Extraer datos en un DataFrame
            dataframe = pd.DataFrame(croissant_dataset.records(record_set=record_sets[0].uuid))
            dataframe.columns = [col.split('/')[-1] for col in dataframe.columns]
            logger.info("Nombres de columnas después de limpieza: %s", dataframe.columns)

            # Replace this with the actual data source URL or path
            data_path = "s3://data/raw/cars.csv"

            # Writing to S3
            wr.s3.to_csv(df=dataframe,
                         path=data_path,
                         index=False)
        except Exception as e:
            print(f"Error occurred: {e}")
            logger.error(e)
            raise

    get_data()


dag = process_el_cars_data()
