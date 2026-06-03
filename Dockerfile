FROM python:3.10

WORKDIR /app/backend

COPY requirements.txt /app/requirements.txt
COPY backend/ .
COPY frontend/ ../frontend

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]