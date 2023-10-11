# api/Dockerfile
FROM python:3.11

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "8000"]
