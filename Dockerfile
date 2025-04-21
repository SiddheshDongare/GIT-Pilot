# Use Python 3.12 slim as base image
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency installation
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies
RUN uv pip install --system .

# Final stage
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 gitpilot && \
    chown -R gitpilot:gitpilot /app

# Switch to non-root user
USER gitpilot

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose FastMCP default port
EXPOSE 8080

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "main.py"]
