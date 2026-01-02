#!/bin/sh
# Simple startup script that respects environment variables.
# Use PORT env var (common on PaaS). Fallback to 8000.
: "${PORT:=8000}"
: "${HOST:=0.0.0.0}"
: "${CHROMA_PERSIST_DIR:=./data/chroma_db}"

# Ensure chroma path exists (helps when a volume is mounted)
mkdir -p "$CHROMA_PERSIST_DIR"

# Print runtime info to logs
echo "Starting app: host=${HOST} port=${PORT} chroma_dir=${CHROMA_PERSIST_DIR}"

# Start uvicorn (1 worker by default; increase if CPU allows)
exec uvicorn app.main:app --host "$HOST" --port "$PORT" --workers 1
