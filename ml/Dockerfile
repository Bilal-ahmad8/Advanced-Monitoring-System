FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy  code
COPY src/ ./src/

# Copy artifacts (model, scaler, etc.)
COPY artifacts/ ./artifacts/

#for model parameter
COPY model_params/ ./model_params/  

#for immediate inferencing 
COPY buffer_data.csv ./buffer_data.csv

# Default command
CMD ["python", "-m", "src.pipeline.inference_pipeline.inference_pipeline"]
