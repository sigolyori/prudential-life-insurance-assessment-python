# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy project files
COPY . .

# Download model and data (if needed)
# RUN python -c "from src.data import download_data; download_data()"

# Expose port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
