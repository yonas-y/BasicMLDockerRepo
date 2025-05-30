FROM ubuntu:latest
FROM python:3.10-slim

LABEL authors="Yonas Y"

#ENTRYPOINT ["top", "-b"]
# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY .. .

# Run the script
CMD ["python", "train_RFC.py"]