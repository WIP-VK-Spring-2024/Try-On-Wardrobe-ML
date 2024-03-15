FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# already has python 3.10.12

# Set the working directory
RUN mkdir /usr/src/app
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python and pip
RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install -U pip && \
    pip3 install poetry

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

COPY pyproject.toml poetry.lock /usr/src/app/

RUN poetry config virtualenvs.create true && \
    poetry install --no-root --no-interaction --no-ansi 

# commented to guarantee sync volume changes
# COPY . /usr/src/app/