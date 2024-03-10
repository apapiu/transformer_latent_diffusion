FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn gunicorn fastapi pytest ruff pytest-asyncio httpx 

EXPOSE 80

CMD ["uvicorn", "tld.app:app", "--host", "0.0.0.0", "--port", "80"]
