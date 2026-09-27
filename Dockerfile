FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV headless still needs libglib.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
RUN pip install -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN pip install -e .

# Non-root user; data, models, outputs and logs are mounted as volumes.
RUN useradd --create-home appuser \
    && mkdir -p /app/data /app/models /app/outputs /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

# Default: API. The dashboard service overrides the command in docker-compose.yml.
CMD ["uvicorn", "cxr_reliability.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
