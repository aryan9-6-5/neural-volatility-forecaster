# Use official light-weight Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies needed for compiling certain packages (e.g. gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to utilize Docker build cache
COPY requirements.txt /app/

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . /app/

# Create raw, historical, and processed data directories
RUN mkdir -p data/raw data/historical data/processed models

# Expose port for FastAPI serving
EXPOSE 8000

# Command to run uvicorn server
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
