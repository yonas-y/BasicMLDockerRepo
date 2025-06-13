# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY app/ app/

# Set environment variable so Python sees /app
ENV PYTHONPATH=/app

# Default command to run your app (main.py should be inside /app)
CMD ["python", "app/main.py"]
