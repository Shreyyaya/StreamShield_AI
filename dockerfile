FROM python:3.10-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir pandas numpy scikit-learn fastapi uvicorn joblib

# Copy all project files
COPY . .

# Expose port
EXPOSE 8050

# Run API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]