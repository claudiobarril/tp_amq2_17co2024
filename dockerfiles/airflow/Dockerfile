FROM apache/airflow:2.10.3

# Switch to root user for installation tasks
USER root

# Update apt-get, install required packages, and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        gcc \
        g++ \
        libatlas-base-dev \
        libopenblas-dev \
        python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -yqq --purge

# Switch back to the airflow user
USER airflow

# Upgrade pip and install dependencies from requirements.txt
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip \
#    && pip install scikit-learn==1.5.2 --index-url https://pypi.org/simple \
    && pip install --no-cache-dir apache-airflow[virtualenv]==2.10.3 -r /requirements.txt

# Optionally, set the working directory
WORKDIR /opt/airflow
