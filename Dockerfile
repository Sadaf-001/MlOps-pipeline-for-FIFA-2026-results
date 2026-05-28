# FIFA 2026 MLOps Pipeline
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system libraries required by LightGBM (used by PyCaret)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and data
COPY src/   ./src/
COPY data/  ./data/

# Create output directories
RUN mkdir -p models logs

# Default command runs the full pipeline in order:
#   1. Engineer time-series features
#   2. Train model via AutoML (saves fifa_model.pkl, encoders.pkl, feature_cols.pkl)
#   3. Fine-tune on same data (smoke test)
#   4. Monitor for drift
#   5. Run test showcase
CMD ["sh", "-c", "\
  python src/feature_engineering.py && \
  python src/train.py && \
  python src/fine_tune.py \
    --new-data data/results_engineered.csv \
    --base-model models/fifa_model.pkl \
    --encoders models/encoders.pkl && \
  python src/monitor.py \
    --reference data/results_engineered.csv \
    --current data/results_engineered.csv \
    --model models/fifa_model.pkl \
    --encoders models/encoders.pkl && \
  python src/test_showcase.py \
    --test-snap data/test_snapshot.csv \
    --base-model models/fifa_model.pkl \
    --ft-model models/fifa_model_finetuned.pkl \
    --encoders models/encoders.pkl \
"]
