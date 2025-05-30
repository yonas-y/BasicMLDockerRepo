# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY ocsvm_ntk_traffic_anomaly.py .
COPY ocsvm_ntk_traffic_fastapi.py .
COPY output/ output/

# Train the model on build (optional)
RUN python ocsvm_ntk_traffic_anomaly.py

# Expose the port for FastAPI
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "ocsvm_ntk_traffic_fastAPI:app", "--host", "0.0.0.0", "--port", "8000"]
