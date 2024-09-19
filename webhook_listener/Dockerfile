FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY webhook_listener.py .

CMD ["python", "webhook_listener.py"]
