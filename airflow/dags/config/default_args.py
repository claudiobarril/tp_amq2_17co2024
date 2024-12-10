import datetime

default_args = {
    'owner': "17co2024",
    'depends_on_past': False,
    'retries': 0,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15),
    'catchup': False,  # Evitar trabajos pendientes
}
