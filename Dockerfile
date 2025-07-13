FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# Set working directory
WORKDIR /app

# Copy code
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port for inference (FastAPI)
EXPOSE 8000

# Default: Run API
CMD ["uvicorn", "infer:app", "--host", "0.0.0.0", "--port", "8000"]