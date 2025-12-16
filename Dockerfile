FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies often required for image processing libraries
# Even with headless, some base libs might be missing in slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set default port
ENV PORT=10000

# Run gunicorn
# Render provides the PORT environment variable
CMD gunicorn app:app --bind 0.0.0.0:$PORT
