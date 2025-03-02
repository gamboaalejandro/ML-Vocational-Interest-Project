# Usar una imagen base de Python 3.11
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar los archivos de requisitos
# Copiar solo el archivo de requisitos primero
COPY requirements.docker.txt ./requirements.txt

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download es_core_news_sm

# Descargar el modelo de spacy
RUN python -m spacy download es_core_news_sm

# Copiar el resto del código
COPY . .

# Crear directorios necesarios si no existen
RUN mkdir -p data/raw data/processed

# Agregar el directorio actual al PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Instalamos JupyterLab y Uvicorn (para la API)
RUN pip install jupyterlab uvicorn fastapi

# Exponemos el puerto 8888 para JupyterLab y 8000 para la API
EXPOSE 8888 8000

# Comando para ejecutar JupyterLab y la API con uvicorn
CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --allow-root & uvicorn src.main:app --host 0.0.0.0 --port 8000"]