FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["sh", "-c", "gunicorn run:app --bind 0.0.0.0:${PORT:-8080} --timeout 300 --workers 1 --max-requests 5 --max-requests-jitter 2"]
