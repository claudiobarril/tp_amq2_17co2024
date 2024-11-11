FROM python:3.12-slim

# Copy your application code
COPY ./poetry.lock .
COPY ./pyproject.toml .
COPY ./tp_amq2_17co2024/ ./tp_amq2_17co2024/

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

RUN poetry config virtualenvs.create false && poetry install

COPY . .

# Expose the desired port
EXPOSE 8800
