# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHURN_MODEL_PATH=/app/artifacts/model.joblib

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-deps .

# Training is deterministic and takes seconds, so the image builds its own model.
# In production the artifact would come from a model registry instead.
RUN python -m churn_service.train --out /app/artifacts > /dev/null

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "churn_service.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
