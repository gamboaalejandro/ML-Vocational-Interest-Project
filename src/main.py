from fastapi import FastAPI, Request, status
from fastapi.openapi.utils import get_openapi
from starlette.responses import JSONResponse
from src.utils.logger import LoggerFactory
from src.features.prediction_router import prediction_router
logger = LoggerFactory.create_logger()


def create_app() -> FastAPI:
    """Crea la aplicación FastAPI con configuración limpia."""
    app = FastAPI(
        swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"},
        title='ML-API-VOCATIONAL-INTEREST',
        debug=False,
        description="API para comunicar el modelo de machine learning con la aplicación principal",
        version="1.0.0",
        contact={
            "name": "Soporte Vocational Interest",
            "email": "agamboacj@gmail.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
    )

    @app.get("/")
    async def root():
        """
        Endpoint raíz que devuelve un mensaje de bienvenida
        """
        return {"message": "Bienvenido a la API de ML Vocational Interest"}

    @app.get("/health")
    async def health_check():
        """
        Endpoint para verificar el estado de la API
        """
        return {"status": "OK", "message": "El servicio está funcionando correctamente"}

    # Personalizar el esquema OpenAPI
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            contact=app.contact,
            license_info=app.license_info,
        )
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi
    return app


logger.log("Iniciando aplicación...")
app = create_app()
app.include_router(prediction_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
