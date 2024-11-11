FROM python:3.8-slim

WORKDIR /app
COPY . .
COPY ../../tp_amq2_17co2024 .
COPY requirements.txt /requirements.txt

RUN apt-get update && apt-get install -y curl
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

EXPOSE 8800
