# ── Cascade — Runtime Adaptive Inference Orchestrator ──────────────────────
# AMD ROCm base with PyTorch pre-installed
FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_2.3.0

LABEL maintainer="Cascade Team"
LABEL description="Closed-loop LLM inference controller — AMD Developer Hackathon ACT II"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (install before copying source for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Runtime environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Model & API config (override at runtime)
ENV FIREWORKS_API_KEY=""
ENV FIREWORKS_MODEL="accounts/fireworks/models/llama-v3p1-70b-instruct"
ENV LOCAL_MODEL_DIR="/app/models"
ENV LOCAL_MODEL_PATH="/app/models/model"

# ROCm GPU config
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV ROCR_VISIBLE_DEVICES=0

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
