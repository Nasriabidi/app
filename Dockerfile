# ============================================================
# Combined Dockerfile - Crack Inspection (AKS Ready)
# ============================================================

# ---- Stage 1: Python API dependencies ----
FROM python:3.10-slim AS api-builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python packages
# CPU-only index used for torch — avoids downloading 2.5 GB of CUDA binaries
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r requirements.txt && \
    # Fix CVE-2026-24049: wheel privilege escalation
    pip install --no-cache-dir "wheel>=0.46.2" && \
    # Fix CVE-2026-23949: jaraco.context path traversal
    pip install --no-cache-dir "jaraco.context>=6.1.0" 

# ---- Stage 2: Final image ----
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system deps + Nginx + supervisor
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed Python packages from builder
COPY --from=api-builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=api-builder /usr/local/bin /usr/local/bin

# Force upgrade vulnerable packages that the base image may have pre-installed
# Fix CVE-2026-24049: wheel privilege escalation (0.45.1 → 0.46.2)
# Fix CVE-2026-23949: jaraco.context path traversal (5.3.0 → 6.1.0)
RUN pip install --no-cache-dir --force-reinstall "wheel==0.46.2" "jaraco.context==6.1.0"

# Copy API source
COPY main.py .
COPY model/ ./model/

# Temp dir for image processing
RUN mkdir -p /app/temp

# Copy frontend static files
COPY webapp/ /usr/share/nginx/html/

# Copy AKS-ready nginx config
COPY nginx.aks.conf /etc/nginx/conf.d/default.conf
RUN rm -f /etc/nginx/sites-enabled/default

# Copy supervisor config (manages nginx + uvicorn)
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# AKS runs as non-root — fix permissions
RUN chown -R www-data:www-data /usr/share/nginx/html && \
    chown -R www-data:www-data /var/log/nginx && \
    chown -R www-data:www-data /var/lib/nginx && \
    chmod -R 755 /usr/share/nginx/html && \
    touch /var/run/nginx.pid && \
    chown www-data:www-data /var/run/nginx.pid && \
    chown -R www-data:www-data /app

# Expose only HTTP — AKS Ingress handles TLS termination
EXPOSE 80
EXPOSE 8000

# Start both services via supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]