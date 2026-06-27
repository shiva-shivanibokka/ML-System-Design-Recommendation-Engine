FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Non-root user for container security
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app
CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
