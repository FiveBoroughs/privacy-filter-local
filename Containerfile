ARG PRIVACY_FILTER_BACKEND=torch
FROM docker.io/python:3.11-slim

ARG PRIVACY_FILTER_BACKEND
ENV HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HUB_DISABLE_SHARED_BLOBS=1 \
    PYTHONUNBUFFERED=1 \
    PRIVACY_FILTER_BACKEND=${PRIVACY_FILTER_BACKEND} \
    LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cu13/lib

WORKDIR /app

COPY requirements.txt requirements-onnx.txt requirements-gliner2.txt ./
RUN if [ "$PRIVACY_FILTER_BACKEND" != "onnx" ]; then \
      apt-get update && \
      apt-get install -y --no-install-recommends gcc libc6-dev && \
      rm -rf /var/lib/apt/lists/*; \
    fi
# gliner2 2.0 declares transformers<5, but this checkpoint's tokenizer requires
# the 5.x format it was saved with. Install the tested package without letting
# that stale cap downgrade the pinned runtime.
RUN if [ "$PRIVACY_FILTER_BACKEND" = "onnx" ]; then \
      pip install --no-cache-dir -r requirements-onnx.txt; \
    elif [ "$PRIVACY_FILTER_BACKEND" = "gliner2" ]; then \
      pip install --no-cache-dir -r requirements-gliner2.txt && \
      pip install --no-cache-dir --no-deps gliner2==2.0.0; \
    else \
      pip install --no-cache-dir -r requirements.txt; \
    fi

COPY scripts/text_chunking.py /app/text_chunking.py
COPY scripts/adaptive_scan.py /app/adaptive_scan.py
COPY scripts/token_classification.py /app/token_classification.py
COPY scripts/onnx_backend.py /app/onnx_backend.py
COPY scripts/gliner2_backend.py /app/gliner2_backend.py
COPY scripts/pplx_model.py /app/pplx_model.py
COPY scripts/pplx_model.LICENSE /app/pplx_model.LICENSE
COPY scripts/pplx_backend.py /app/pplx_backend.py
COPY scripts/privacy_filter_service.py /app/privacy_filter_service.py

EXPOSE 8757

CMD ["uvicorn", "privacy_filter_service:app", "--host", "0.0.0.0", "--port", "8757"]
