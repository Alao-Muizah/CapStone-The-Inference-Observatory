FROM python:3.10

WORKDIR /app/backend

COPY backend/ .
COPY frontend/ ../frontend

RUN pip install fastapi uvicorn

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]