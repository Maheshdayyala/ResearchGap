FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .

# Install CPU-only PyTorch (avoid huge NVIDIA CUDA downloads)
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

# Install remaining project dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt


COPY . .
EXPOSE 5000
# one process, many threads: the background pipeline thread-pool lives inside the web process
CMD ["gunicorn", "-w", "1", "--threads", "8", "--timeout", "180", "-b", "0.0.0.0:5000", "wsgi:app"]
