# ============================================================
# Fractal — Production Dockerfile (Hardened Multi-Stage Build)
# Base: python:3.11-slim with security hardening
# ============================================================

# ── Stage 1: Builder ──
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Node.js dependencies (root package)
COPY package.json package-lock.json* ./
RUN npm ci --ignore-scripts 2>/dev/null || npm install --ignore-scripts

# Trigger.dev jobs dependencies
COPY jobs/package.json jobs/tsconfig.json ./jobs/
RUN cd jobs && (npm ci --ignore-scripts 2>/dev/null || npm install --ignore-scripts)


# ── Stage 2: Runtime (Hardened) ──
FROM python:3.11-slim AS runtime

LABEL maintainer="Fractal Team" \
      description="Fractal Agentic Infrastructure — Hardened Runtime" \
      version="1.0.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/node_modules/.bin:${PATH}"

# Install minimal runtime dependencies + Node.js
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    iptables \
    tini \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# ── Security Hardening ──
# Remove unnecessary setuid/setgid binaries
RUN find / -perm /6000 -type f -exec chmod a-s {} + 2>/dev/null || true

# Remove package managers to prevent runtime tampering
RUN rm -f /usr/bin/apt-get /usr/bin/dpkg /usr/bin/apt 2>/dev/null || true

# Create non-root user
RUN groupadd --gid 1001 fractal \
    && useradd --uid 1001 --gid fractal --shell /bin/false --create-home fractal

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /install /usr/local

# Copy Node.js modules from builder
COPY --from=builder /build/node_modules ./node_modules
COPY --from=builder /build/jobs/node_modules ./jobs/node_modules

# Copy application code
COPY --chown=fractal:fractal . .

# Make scripts executable
RUN chmod +x scripts/*.sh 2>/dev/null || true

# Create required directories with correct ownership
RUN mkdir -p /app/data /app/logs /app/models /app/certs \
    && chown -R fractal:fractal /app/data /app/logs /app/models /app/certs

# ── Health check ──
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Drop to non-root user
USER fractal

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["tini", "--"]

# Default command: start the orchestrator
CMD ["python", "-m", "src.agents.task_agent"]

EXPOSE 8000
