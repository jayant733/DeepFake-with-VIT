# Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set environment variables for optimized Python
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=2

# Copy project files
COPY . .

# Expose port for HF Spaces
EXPOSE 7860

# Run app with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers=1", "--timeout=60", "app:app"]
