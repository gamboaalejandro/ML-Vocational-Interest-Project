# Usar una imagen base más reciente de NVIDIA
FROM nvidia/cuda:12.3.1-base-ubuntu22.04

# Configurar variables de entorno para CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Instalar Python y dependencias necesarias
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de requisitos
COPY requirements.docker.txt ./requirements.txt

# Instalar dependencias de Python
RUN pip3 install --no-cache-dir -r requirements.txt \
    && python3 -m spacy download es_core_news_sm

# Crear directorios necesarios si no existen
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/notebooks

# Copiar el resto del código manteniendo la estructura
COPY . /app/

# Agregar el directorio actual al PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Crear directorio específico para modelos si no existe
RUN mkdir -p /app/models

# Instalamos JupyterLab y Uvicorn (para la API)
RUN pip3 install jupyterlab uvicorn fastapi email-validator keybert intel-extension-for-pytorch


# Exponemos el puerto 8888 para JupyterLab y 8000 para la API
EXPOSE 8888 8000

# Comando para ejecutar JupyterLab y la API con uvicorn
CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --allow-root & uvicorn src.main:app --host 0.0.0.0 --port 8000"]