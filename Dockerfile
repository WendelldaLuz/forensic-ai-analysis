FROM python:3.11-slim

LABEL maintainer="Wendell da Luz Silva"
LABEL description="Sistema de IA Forense Evolutiva"

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data logs models config security

EXPOSE 8501

CMD ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
