FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir fastapi uvicorn httpx numpy pydantic

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV FIREWORKS_API_KEY=""
ENV FIREWORKS_MODEL="accounts/fireworks/models/llama-v3p1-70b-instruct"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
