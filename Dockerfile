# Builder stage with full dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements only
COPY requirements.txt .

# Install Python packages into /install
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt

# Copy your app files
COPY . .

# Final stage: smaller runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy app source code
COPY . .

# Expose ports
EXPOSE 5000
EXPOSE 8501

# Run your app (Flask or Streamlit)
CMD ["python", "app.py"]
# For Streamlit, comment above and uncomment below
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
