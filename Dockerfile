FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-bake the embedding model so startup is faster
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

# On first start: downloads Kaggle data + builds ChromaDB + uploads to Drive
# On later starts: restores ChromaDB from Drive (fast)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]