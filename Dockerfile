# ==========================================
# STAGE 1: Build the React Frontend
# ==========================================
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ==========================================
# STAGE 2: Run the FastAPI Backend & Serve UI
# ==========================================
FROM python:3.10-slim AS runner
WORKDIR /app

# Install system dependencies (needed for compiling python packages if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install them
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy ML saved models (inference engine needs this)
COPY ml/saved_models/cardiac_risk_model.pkl ./ml/saved_models/cardiac_risk_model.pkl

# Copy outputs folder needed by history.py for correlation csv file
COPY outputs/correlation_percentage_table.csv ./outputs/correlation_percentage_table.csv

# Copy backend application source
COPY backend/app ./backend/app

# Copy the compiled frontend build from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port and start FastAPI server
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
