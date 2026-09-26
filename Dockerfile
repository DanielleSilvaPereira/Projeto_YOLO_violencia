FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# torch CPU (imagem menor); para GPU use uma base nvidia/cuda
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY ia ia
COPY servico servico
COPY pesos pesos
ENV PORT=8090
EXPOSE 8090
CMD ["sh", "-c", "uvicorn servico.app:app --host 0.0.0.0 --port ${PORT}"]
