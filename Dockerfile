FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends make && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Data is not in the repo: mount it at runtime, e.g.
#   docker run -p 8000:8000 -v $PWD/data:/app/data apex-copilot   (no API key needed: the service is fully local)
# On first start the knowledge base and classifier are built if missing.
EXPOSE 8000
CMD ["sh", "-c", "[ -f models/kb.pkl ] || python scripts/build_models.py; uvicorn app.main:app --host 0.0.0.0 --port 8000"]
