from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings
import os
from pathlib import Path

class Settings(BaseSettings):
    APP_NAME: str = "ML Vocational Interest API"  # valor por defecto
    ENVIRONMENT: str = "dev"  # default environment
    DEBUG: bool = True
    SECRET_KEY: str = "tu_clave_secreta"
    ALLOWED_ORIGINS: str = "*"
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/db"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        # Obtener el ambiente del entorno o usar 'dev' por defecto
        env = os.getenv("ENVIRONMENT", "dev")

        # Obtener la ruta absoluta al directorio actual
        current_dir = Path(__file__).parent
        
        # Construir la ruta al archivo .env según el ambiente
        env_files = {
            "dev": current_dir / "deploy" / ".env.dev",
            "prod": current_dir / "deploy" / ".env.prod",
            "test": current_dir / "deploy" / ".env.test"
        }
        
        # Seleccionar el archivo .env correspondiente
        env_file = str(env_files.get(env))
        
        # Verificar si el archivo existe
        if not Path(env_file).exists():
            raise FileNotFoundError(f"Archivo .env.{env} no encontrado en: {env_file}")
            
        env_file_encoding = "utf-8"


@lru_cache
def get_environment_variables():
    return Settings()
