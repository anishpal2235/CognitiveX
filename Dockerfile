FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so code edits don't invalidate the pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persist the trace store and audit chain outside the container. An audit log
# that disappears on redeploy is not an audit log.
ENV DB_PATH=/var/lib/controlplane/controlplane.db
RUN mkdir -p /var/lib/controlplane
VOLUME ["/var/lib/controlplane"]

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

# Single worker by default: the bandit state and session risk window are
# per-process. See README > Running on your own server.
CMD ["uvicorn", "controlplane.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
