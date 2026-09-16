FROM docker.io/python:3.11-slim

ENV HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    PYTHONUNBUFFERED=1 \
    PRIVACY_FILTER_MODEL=openai/privacy-filter

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/text_chunking.py /app/text_chunking.py
COPY scripts/adaptive_scan.py /app/adaptive_scan.py
COPY scripts/privacy_filter_service.py /app/privacy_filter_service.py

EXPOSE 8757

CMD ["uvicorn", "privacy_filter_service:app", "--host", "0.0.0.0", "--port", "8757"]
