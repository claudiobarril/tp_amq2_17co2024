FROM python:3.11-slim

WORKDIR /app

# Copy files
COPY . .
COPY ../../poetry.lock .
COPY ../../pyproject.toml .
COPY ../../tp_amq2_17co2024/ .

RUN apt-get clean && apt-get update --allow-insecure-repositories && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false && poetry install

EXPOSE 8800
EXPOSE 8501

CMD bash -c "poetry run uvicorn app:app --host 0.0.0.0 --port 8800 & poetry run streamlit run front.py --server.port 8501 --server.address 0.0.0.0"
