FROM python:3.11-slim

WORKDIR /app

# System dependencies needed by scikit-learn/xgboost wheels and SQLite
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/, models/ are expected to be mounted as volumes (see docker-compose.yml)
# so the evaluator's downloaded dataset and trained model are used directly
# rather than baked into the image.
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
