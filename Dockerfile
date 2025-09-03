FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./conf /app/conf
COPY utils.py .
COPY 04_inference.py .
COPY loader_redis.py .

ENV APPCONF=conf/app.yaml

CMD ["uvicorn","04_inference:app","--host","0.0.0.0","--port","8123", "--workers","1"]
