#!/bin/bash

# Iniciar JupyterLab en segundo plano
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' &

# Iniciar la API
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload 