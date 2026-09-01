# CPU-only image. Nothing in this project uses a GPU, and the CUDA build of
# PyTorch is roughly ten times larger.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models

WORKDIR /app

# Dependencies first, so a code change does not invalidate the layer that took
# several minutes to build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding and reranker weights into the image. This adds a few hundred
# MB but means the first question is answered in seconds rather than after a
# ~170 MB download, and the container works with no network at all.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY ragbot/ ./ragbot/
COPY eval/ ./eval/
COPY samples/ ./samples/
COPY scripts/ ./scripts/
COPY app.py ./

# Index lives on a volume so it survives container replacement.
ENV RAGBOT_INDEX_PATH=/data/index
VOLUME /data
RUN mkdir -p /data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Run as a non-root user. /opt/models must be owned by that user too: the
# huggingface cache writes lock files on load, and a root-owned cache would
# fail at runtime rather than at build time.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /data /app /opt/models
USER appuser

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
