FROM python:3.10

WORKDIR /app

COPY backend/ backend/
COPY frontend/ frontend/
COPY backend/services/ backend/services/

RUN pip install fastapi uvicorn

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
