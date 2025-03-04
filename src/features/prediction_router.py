from fastapi import APIRouter, status, Depends
from typing import List
from pydantic import BaseModel
from src.features.model_utils import (
    predict_career_with_keywords,
    preprocess_text,
    tokenizer,
    model,
    index_to_category,
    device,
)

# DTO para la solicitud: cada objeto debe tener la clave 'respuesta'
class AnswerInput(BaseModel):
    respuesta: str

# DTO para las predicciones individuales
class PredictionDetail(BaseModel):
    career: str
    probability: float

# DTO para la salida de la predicción de cada texto
class PredictionOutput(BaseModel):
    original_text: str
    preprocessed_text: str
    predictions: List[PredictionDetail]

# DTO para encapsular la respuesta final
class PredictionsResult(BaseModel):
    results: List[PredictionOutput]

# Crear el router con el prefijo y las etiquetas que prefieras
prediction_router = APIRouter(
    prefix="/predictions",
    tags=["Predicciones"]
)

@prediction_router.post(
    "/",
    response_model=PredictionsResult,
    status_code=status.HTTP_200_OK,
    #dependencies=[Depends(JWTBearer())],
    responses={
        200: {"description": "Predicciones obtenidas exitosamente"},
        400: {"description": "Solicitud incorrecta"},
        401: {"description": "No autorizado"},
        500: {"description": "Error interno del servidor"},
    },
)
async def get_predictions(inputs: List[AnswerInput]):
    """
    **Realiza predicciones de carreras a partir de las respuestas suministradas.**
    
    Recibe un arreglo de objetos JSON en los que cada objeto debe tener la clave `respuesta`
    y retorna, para cada respuesta, el texto original, su versión preprocesada y las predicciones
    (carrera y probabilidad) obtenidas.
    """
    # Extraer la lista de textos
    texts = [item.respuesta for item in inputs]
    
    predictions_list = predict_career_with_keywords(
    text=texts,
    tokenizer=tokenizer,
    model=model,
    index_to_category=index_to_category,
    device=device,
    temperature=1.3,
    top_k=3
    )
    
    # Construir la respuesta
    results = []
    for original_text, preds in zip(texts, predictions_list):
        preprocessed = preprocess_text(original_text)
        preds_details = [{"career": career, "probability": prob} for career, prob in preds]
        results.append({
            "original_text": original_text,
            "preprocessed_text": preprocessed,
            "predictions": preds_details
        })
    
    return {"results": results} 