FROM python:3.11-slim

# keep Python from writing .pyc files and make output stream unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

# System deps needed by many ML and audio libs and for building wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       curl \
       ffmpeg \
       libsndfile1 \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install python deps (no-cache-dir to keep image small)
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY . /app

# Ensure start script is executable
RUN chmod +x /app/start.sh

# Create a non-root user and set ownership
RUN addgroup --system app && adduser --system --ingroup app app \
    && chown -R app:app /app

USER app

# Default port (overridden by PORT env var if set)
EXPOSE 8000

# Entrypoint script will use PORT env var if provided
CMD ["./start.sh"]
