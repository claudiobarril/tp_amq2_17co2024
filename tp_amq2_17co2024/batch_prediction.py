import logging
import pandas as pd

from datetime import datetime
from botocore.exceptions import ClientError


def add_request_to_file(client, data, bucket_name, file_name, file_folder):
    file_local_path = '/tmp/' + file_name
    try:
        client.download_file(bucket_name, file_folder + '/' + file_name, file_local_path)
        df = pd.read_csv(file_local_path)
        df = pd.concat([df, data], ignore_index=True)
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            df = data
        else:
            raise e
    except Exception as e:
        raise
    df.to_csv(file_local_path, index=False)
    client.upload_file(file_local_path, bucket_name, file_folder + '/' + file_name)


def add_request(client, data, bucket_name="data", file_name="cars_X_to_predict.csv", file_folder="final"):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Obtener la fecha de hoy en formato YYYY-MM-DD y la agregamos como prefijo
    today_date = datetime.now().strftime('%Y-%m-%d')
    daily_file_name = f"{today_date}_{file_name}"

    logger.info(f"Adding new prediction request")
    add_request_to_file(client, data, bucket_name, file_name, file_folder)
    add_request_to_file(client, data, bucket_name, daily_file_name, file_folder)

    logger.info(f"new prediction request added")
