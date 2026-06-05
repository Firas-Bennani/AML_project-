FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install torch-geometric \
 && pip install -r requirements.txt

COPY . /app/

EXPOSE 8001

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=6 \
  CMD curl -fsS http://localhost:8001/healthz || exit 1

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8001"]
