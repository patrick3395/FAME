# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed by scipy/matplotlib
RUN apt-get update     && apt-get install -y --no-install-recommends        build-essential        libatlas-base-dev        gfortran        libjpeg-dev        zlib1g-dev     && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

ENV PORT=8080

CMD ["gunicorn", "--bind", "0.0.0.0:", "FAME_UI:app"]
