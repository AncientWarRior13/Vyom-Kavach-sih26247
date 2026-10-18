# Stage 1: Build frontends
FROM node:20-alpine AS frontend-builder
WORKDIR /workspace

# Copy design tokens if present
COPY packages/ ./packages/

# Build tactical app
COPY app/package*.json ./app/
RUN cd app && npm install
COPY app/ ./app/
RUN cd app && npm run build

# Build landing page
COPY landing/package*.json ./landing/
RUN cd landing && npm install
COPY landing/ ./landing/
RUN cd landing && npm run build

# Stage 2: Python runtime
FROM python:3.12-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    DRONETHREATTRAINER_ALLOW_REMOTE=1 \
    PORT=8090

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy built frontend distributions from stage 1
COPY --from=frontend-builder /workspace/app/dist ./app/dist
COPY --from=frontend-builder /workspace/landing/dist ./landing/dist

EXPOSE 8090

CMD ["sh", "-c", "python -m uvicorn server:app --host 0.0.0.0 --port ${PORT:-8090}"]
