# %% [markdown]
# ## Preparar el entorno y carga del modelo 

# %%
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Mapea índice -> categoría (lo que usaste en training)
index_to_category = {
    0: "INDUSTRIAL", 
    1: "CIVIL", 
    2: "INFORMÁTICA",
    3: "TELECOMUNICACIONES",
    4: "ARQUITECTURA",
    5: "FILOSOFÍA",
    6: "PSICOLOGÍA",
    7: "LETRAS",
    8: "COMUNICACIÓN SOCIAL",
    9: "EDUCACIÓN",
    10: "ADMINISTRACIÓN",
    11: "CONTADURÍA",
    12: "RELACIONES INDUSTRIALES",
    13: "SOCIOLOGÍA",
    14: "ECONOMÍA",
    15: "DERECHO",
    16: "TEOLOGÍA"
}


# Dispositivo (GPU si está disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar tokenizer
tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')

# Cargar modelo con la misma configuración
model = BertForSequenceClassification.from_pretrained(
    'dccuchile/bert-base-spanish-wwm-cased',
    num_labels=len(index_to_category)
)
model.load_state_dict(torch.load('notebooks/best_model_state.bin', map_location=device))
model.to(device)
model.eval()  # Modo inferencia

# %%
## predict carrer

def predict_career(text_list, tokenizer, model, max_length=128):
    """
    text_list: lista de strings con las respuestas de usuario
    tokenizer: tokenizer de Hugging Face
    model: modelo BertForSequenceClassification cargado
    """
    # Tokenizar
    inputs = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, num_labels]
    
    # Obtener predicciones
    preds = torch.argmax(logits, dim=1).cpu().numpy()  # Índices de clase
    predicted_labels = [index_to_category[p] for p in preds]
    return predicted_labels



def predict_career_top3(text_list, tokenizer, model, index_to_category, 
                        device, max_length=512, top_k=3, threshold=0.5, temperature=2.0):
    """
    text_list: lista de strings con las respuestas de usuario
    tokenizer: tokenizer de Hugging Face
    model: modelo BertForSequenceClassification cargado
    index_to_category: diccionario {índice: categoría}
    device: 'cuda' o 'cpu'
    max_length: longitud máxima de tokenización
    top_k: número de predicciones (top) a retornar
    threshold: umbral mínimo de probabilidad para asignar una clase (vs 'UNKNOWN')
    """
    # Tokenizar
    inputs = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Aplicar temperatura para suavizar las probabilidades
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        
        top_probs, top_indices = probs.topk(top_k, dim=1, largest=True, sorted=True)

    batch_results = []
    for i in range(len(text_list)):
        # Extraer las k probabilidades e índices para esta fila i
        row_probs = top_probs[i].cpu().numpy()
        row_indices = top_indices[i].cpu().numpy()

        result = []
        for idx, p in zip(row_indices, row_probs):
            category_name = index_to_category[idx]
            result.append((category_name, float(p)))
        batch_results.append(result)

    return batch_results


import spacy
import re

# Cargar el modelo de SpaCy para español
nlp = spacy.load("es_core_news_sm")

def preprocess_text(text: str) -> str:
    """
    Preprocesa el texto aplicando los siguientes pasos:
    1. Convierte a minúsculas
    2. Elimina caracteres especiales manteniendo acentos
    3. Elimina espacios extra
    4. Elimina stopwords
    5. Aplica lematización
    
    Args:
        text (str): Texto a preprocesar
        
    Returns:
        str: Texto preprocesado
    """
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar caracteres especiales pero mantener acentos
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)
    
    # Reemplazar múltiples espacios por uno solo
    text = re.sub(r'\s+', ' ', text)
    
    # Eliminar espacios al inicio y final
    text = text.strip()
    
    # Procesar con SpaCy para lematización y eliminación de stopwords
    doc = nlp(text)
    
    # Unir los lemas de las palabras que no son stopwords ni puntuación
    cleaned_text = " ".join([token.lemma_ for token in doc 
                           if not token.is_stop and not token.is_punct])
    
    return cleaned_text

def predict_career_with_preprocessing(text_list: list, tokenizer, model, 
                                    index_to_category: dict, device: str,
                                    max_length: int = 512, top_k: int = 3, temperature: float = 2.0) -> list:
    """
    Preprocesa el texto y realiza la predicción de carreras.
    
    Args:
        text_list (list): Lista de textos a procesar
        tokenizer: Tokenizer de BERT
        model: Modelo BERT entrenado
        index_to_category (dict): Mapeo de índices a nombres de carreras
        device (str): Dispositivo para procesamiento ('cuda' o 'cpu')
        max_length (int): Longitud máxima de tokens
        top_k (int): Número de predicciones top a retornar
        
    Returns:
        list: Lista de tuplas (carrera, probabilidad) para cada texto
    """
    # Concatenar todas las respuestas en un solo texto
    combined_text = " ".join(text_list)

    # Preprocesar el texto combinado
    processed_text = preprocess_text(combined_text)
    print("tesxcto", processed_text)
    # Tokenizar
    inputs = tokenizer(
        [processed_text],  # Ahora es una lista con un solo texto
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Aplicar temperatura para suavizar las probabilidades
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        
        top_probs, top_indices = probs.topk(top_k, dim=1, largest=True, sorted=True)

    # Procesar resultados (ahora solo hay una predicción)
    row_probs = top_probs[0].cpu().numpy()
    row_indices = top_indices[0].cpu().numpy()

    result = []
    for idx, p in zip(row_indices, row_probs):
        category_name = index_to_category[idx]
        result.append((category_name, float(p)))

    return [result]  # Mantener la estructura de retorno como lista de listas

# Ejemplo de uso:
user_text = """Este es un texto el cual se debe intentar procesar Talento nato para programar """

# Realizar predicción con el texto preprocesado
predictions = predict_career_with_preprocessing(
    text_list=[user_text],
    tokenizer=tokenizer,
    model=model,
    index_to_category=index_to_category,
    device=device,
    top_k=3
)

print("Texto original:")
print(user_text)
print("\nTexto preprocesado:")
print(preprocess_text(user_text))
print("\nPredicciones:")
for career, prob in predictions[0]:
    print(f"{career}: {prob:.2%}")


