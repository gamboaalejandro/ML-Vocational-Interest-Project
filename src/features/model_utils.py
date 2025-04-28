import torch
from transformers import BertTokenizer, BertForSequenceClassification
from keybert import KeyBERT
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from pathlib import Path

from src.utils.logger import LoggerFactory
logger = LoggerFactory.create_logger()

# 📌 Mapea índice -> categoría
index_to_category = {
    0: "INDUSTRIAL", 1: "CIVIL", 2: "INFORMÁTICA", 3: "TELECOMUNICACIONES",
    4: "ARQUITECTURA", 5: "FILOSOFÍA", 6: "PSICOLOGÍA", 7: "LETRAS",
    8: "COMUNICACIÓN SOCIAL", 9: "EDUCACIÓN", 10: "ADMINISTRACIÓN",
    11: "CONTADURÍA", 12: "RELACIONES INDUSTRIALES", 13: "SOCIOLOGÍA",
    14: "ECONOMÍA", 15: "DERECHO", 16: "TEOLOGÍA"
}

# Obtener la ruta base del proyecto y la ruta al modelo
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / 'notebooks' / 'best_model_state.bin'

# Dispositivo (GPU si está disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar tokenizer
tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')

# Cargar modelo con la misma configuración
try:
    model = BertForSequenceClassification.from_pretrained(
        'dccuchile/bert-base-spanish-wwm-cased',
        num_labels=len(index_to_category)
    )
    
    try:
        if MODEL_PATH.exists():
            model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
            print(f"Modelo cargado exitosamente desde {MODEL_PATH}")
        else:
            print(f"Advertencia: No se encontró el archivo en {MODEL_PATH}. Usando modelo base sin fine-tuning.")
            print(f"Ruta actual del script: {CURRENT_DIR}")
            print(f"Ruta base del proyecto: {PROJECT_ROOT}")
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}. Usando modelo base sin fine-tuning.")
    
    model.to(device)
    model.eval()  # Modo inferencia
except Exception as e:
    print(f"Error crítico al inicializar el modelo: {str(e)}")
    model = None

# 📌 Cargar modelo de KeyBERT
kw_model = KeyBERT("sentence-transformers/bert-base-nli-mean-tokens")

# 📌 Cargar modelo de SpaCy para español
nlp = spacy.load("es_core_news_sm")
stopwords_es = list(nlp.Defaults.stop_words)



####FUNCTIONS FOR TEST PONDERATION

# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^\w\sáéíóúñü]', '', text)  # Eliminar caracteres especiales, pero conservar tildes
#     text = re.sub(r'\s+', ' ', text).strip()  # Reemplaza múltiples espacios
    
#     # Procesar el texto con SpaCy sin eliminar todas las stopwords
#     doc = nlp(text)
#     cleaned_text = " ".join([token.text for token in doc if not token.is_punct])
    
#     return cleaned_text


def preprocess_text(text):
    # Si es una lista, unir los elementos en un solo string
    if isinstance(text, list):
        text = " ".join(text)
    
    text = text.lower()
    text = re.sub(r'[^\w\sáéíóúñü]', '', text)  # Eliminar caracteres especiales, pero conservar tildes
    text = re.sub(r'\s+', ' ', text).strip()  # Reemplaza múltiples espacios
    
    # Procesar el texto con SpaCy sin eliminar todas las stopwords
    doc = nlp(text)
    cleaned_text = " ".join([token.text for token in doc if not token.is_punct])
    
    return cleaned_text

def extract_keywords(text, top_k=10):
    """
    Extrae palabras clave utilizando KeyBERT y TF-IDF.
    """
    
    print("el top k es ", top_k)
    clean_text = preprocess_text(text)
    keybert_keywords = kw_model.extract_keywords(clean_text, keyphrase_ngram_range=(1, 1), stop_words=stopwords_es, top_n=top_k)
    vectorizer = TfidfVectorizer(stop_words=stopwords_es)
    tfidf_scores = vectorizer.fit_transform([clean_text]).toarray().flatten()
    tfidf_tokens = vectorizer.get_feature_names_out()

    # Crear un diccionario para mantener el score más alto para cada palabra
    keyword_scores = {}
    
    # Agregar scores de KeyBERT
    for word, score in keybert_keywords:
        keyword_scores[word] = score
        
    # # Agregar o actualizar con scores de TF-IDF
    # for idx, token in enumerate(tfidf_tokens):
    #     if token in keyword_scores:
    #         keyword_scores[token] = max(keyword_scores[token], tfidf_scores[idx])
    #     else:
    #         keyword_scores[token] = tfidf_scores[idx]
            
    # Agregar/actualizar con scores de TF-IDF
    for idx, token in enumerate(tfidf_tokens):
        normal_token = token.strip().lower()
        if normal_token in keyword_scores:
            # Mantenemos el score máximo
            keyword_scores[normal_token] = max(keyword_scores[normal_token], tfidf_scores[idx])
        else:
            keyword_scores[normal_token] = tfidf_scores[idx]

    # Ordenar por score y tomar los top_k únicos
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)

    unique_keywords = []
    seen_words = set()

    for word, score in sorted_keywords:
        # 'word' ya está normalizada (sin espacios ni mayúsculas)
        if word not in seen_words:
            seen_words.add(word)
            unique_keywords.append((word, score))
            if len(unique_keywords) == top_k:
                break

    print("unique keywords", unique_keywords)
    return unique_keywords


def predict_career_with_keywords(text, tokenizer, model, index_to_category, device, max_length=512, top_k=15, temperature=1.0):
    """
    Preprocesa el texto, extrae palabras clave y realiza la predicción de carreras.
    """
    print("ENTRANDO A KEYBERT ", text)
    keywords = extract_keywords(text, top_k=20)
    extracted_keywords = " ".join([word for word, _ in keywords])
    print("ENTRANDO A KEYBERT ")
    # 🔹 Concatenar palabras clave al texto original
    enriched_text = f"{text} {extracted_keywords}"
    logger.log(f"Texto con palabras clave: {enriched_text}", "info")
    print(enriched_text)
    
    inputs = tokenizer(
        [enriched_text],  
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
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        top_probs, top_indices = probs.topk(top_k, dim=1, largest=True, sorted=True)

    batch_results = []
    for i in range(len([text])):
        row_probs = top_probs[i].cpu().numpy()
        row_indices = top_indices[i].cpu().numpy()

        result = []
        for idx, p in zip(row_indices, row_probs):
            category_name = index_to_category[idx]
            result.append((category_name, float(p)))
        batch_results.append(result)

    return batch_results


def predict_career_with_preprocessing(text_list: list, tokenizer, model, 
                                   index_to_category: dict, device: str,
                                   max_length: int = 128, top_k: int = 3, temperature: float = 1.0) -> list:
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
    # Preprocesar cada texto
    processed_texts = [preprocess_text(text) for text in text_list]
    
    # Tokenizar
    inputs = tokenizer(
        processed_texts,
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
        row_probs = top_probs[i].cpu().numpy()
        row_indices = top_indices[i].cpu().numpy()

        result = []
        for idx, p in zip(row_indices, row_probs):
            category_name = index_to_category[idx]
            result.append((category_name, float(p)))
        batch_results.append(result)

    return batch_results

# # Ejemplo de uso:
# user_text = """Como desarrollador de IA, creo modelos para luego utilizarlos en aplicaciones. 
# Me gustaba mucho la parte matemática, por ser algo práctico una vez entendías el tema. 
# Sin embargo, también me gustaba la materia de literatura, historia y biología. 
# Educación física, soy mala en los deportes. 
# Si, un curso de inglés, me gusta mucho el idioma. 
# Me gusta ver series y películas, también ver documentales o videos para aprender cosas nuevas de ciencia, historia, cultura general. 
# Ciencia, tecnología, arte y cultura. 
# Acerca de ciencia, como la teoría de la relatividad, cuántica, me gusta mucho entender el mundo desde esa perspectiva. 
# Crear, investigar y resolver problemas. 
# Me interesa entender cómo funciona el mundo, Me gusta imaginar y crear cosas nuevas, Prefiero resolver problemas prácticos y concretos. 
# En un café. 
# Tecnológico. 
# Emprender."""

# # user_text = """Desde muy joven, he sentido una gran pasión por la tecnología. Me encanta aprender y desarrollar soluciones de software que resuelvan problemas reales; por ello, he orientado mis estudios hacia la Ingeniería Informática. Además, me fascina el mundo de las telecomunicaciones, ya que creo que conectar a las personas a través de redes modernas y eficientes es clave para el avance social y económico. Por otro lado, también me interesa la parte gerencial y organizativa, lo que me lleva a valorar la Administración; considero esencial saber planificar, gestionar proyectos y liderar equipos para llevar adelante iniciativas tecnológicas. En resumen, mi formación y mis intereses se centran en el desarrollo de sistemas informáticos, la conectividad a través de telecomunicaciones y la gestión estratégica en entornos empresariales."""

# #user_text = """Estoy interesado en Ingeniería Industrial. Fui becado y congelé mi carrera en 2020 por temas económicos. Me gustaban las materias de Matemática e Historia de Venezuela. No me agradaban Biología y Física. Me interesa resolver problemas y explorar culturas. Me gustaría ejercer mi carrera en España y viajar por el mundo."""

# # Realizar predicción con el texto preprocesado
# predictions = predict_career_with_preprocessing(
#     text_list=[user_text],
#     tokenizer=tokenizer,
#     model=model,
#     index_to_category=index_to_category,
#     device=device,
#     temperature=1.3,
#     top_k=3,
# )

# print("Texto original:")
# print(user_text)
# print("\nTexto preprocesado:")
# print(preprocess_text(user_text))
# print("\nPredicciones:")
# for career, prob in predictions[0]:
#     print(f"{career}: {prob:.2%}")


