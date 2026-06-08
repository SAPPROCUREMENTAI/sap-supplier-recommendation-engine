# ── Supplier Recommendation Engine ─────────────────────────────────────────────
# SAP BTP AI Core compatible Docker image
# Author: Nikhil Bhosale / ProcureArc

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY api/app.py         ./app.py
COPY data/              ./data/
COPY model/             ./model/

# SAP AI Core expects the server on port 8080
EXPOSE 8080

# Use gunicorn for production serving
# SAP AI Core health check hits /health
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "app:app"]
