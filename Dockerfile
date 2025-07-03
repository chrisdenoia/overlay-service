FROM python:3.10-slim

# Install system-level dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables (optional, but can reduce warnings)
ENV PYTHONUNBUFFERED=1

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app

# Run your MediaPipe overlay service
CMD ["python", "overlay-service.py"]