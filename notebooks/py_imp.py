# %% [markdown]
# ## Declare variables for data routes

# %%
DATASET_PATH = '../data/raw/data_carrers.csv'
OUTPUT_PATH = '../data/processed/processed-dataset.csv'

# %%
import pandas as pd

# Cargar el dataset
df = pd.read_csv(DATASET_PATH, encoding="UTF-8")

# Mostrar las primeras filas para ver qué datos tenemos
df.head()

# %% [markdown]
# ## Cleaning data and verify cells

# %%
# Comprobar si hay valores nulos
df.isnull().sum()

df = df.dropna()

#eliminar filas duplicadas
df = df.drop_duplicates()

# %% [markdown]
# # transformando datos para analisis

# %%
categories = df['CARRERA'].unique().tolist()

print(categories)


# %%
category_to_index = {category: idx for idx, category in enumerate(categories)}

print(category_to_index)

# %%
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
from sklearn.model_selection import train_test_split

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# %%
texts = df['TEXTO'].tolist()
print (texts)

labels = df['CARRERA'].map(category_to_index).tolist()  # Mapea las categorías a índices numéricos

print(labels)   

inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

# Obtener las entradas tokenizadas
input_ids = inputs['input_ids']

attention_mask = inputs['attention_mask']

# Convertir las etiquetas (labels) a tensor de PyTorch
labels = torch.tensor(labels)

print('labels tensor',labels)

dataset = TensorDataset(input_ids, attention_mask, labels) # creacion de datasetObject 

# Dividir el dataset en conjunto de entrenamiento y prueba
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Crear DataLoader para cargar los datos
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16)


from transformers import BertForSequenceClassification

# Cargar el modelo preentrenado de BERT para clasificación de secuencias
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(category_to_index))

# Ver el modelo cargado
print(model)

optimizer = AdamW(model.parameters(), lr=2e-5)  # Tasa de aprendizaje 2e-5, comúnmente usada con BERT


# %% [markdown]
# # Entrenamiento del modelo 

# %%
# Definir el número de épocas (epochs) para entrenar
epochs = 3

# Configurar el modelo en modo de entrenamiento
model.train()

# Entrenar durante 3 épocas
for epoch in range(epochs):
    total_loss = 0
    for batch in train_dataloader:
        # Limpiar los gradientes previos
        optimizer.zero_grad()
        
        # Desempaquetar el batch
        input_ids, attention_mask, labels = batch
        
        # Pasar los datos a través del modelo
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        # Obtener la pérdida (loss)
        loss = outputs.loss
        total_loss += loss.item()
        
        # Retropropagación (backpropagation)
        loss.backward()
        
        # Actualizar los pesos
        optimizer.step()

    # Imprimir la pérdida promedio por época
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")


# %% [markdown]
# ## Evaluacion del modelo 

# %%
from sklearn.metrics import classification_report

# Poner el modelo en modo de evaluación
model.eval()

# Inicializar las listas para almacenar las predicciones y las etiquetas verdaderas
predictions = []
true_labels = []

# Evaluar el modelo sin calcular gradientes
with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch
        
        # Realizar la predicción
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Obtener la predicción más probable (la que tiene el valor más alto)
        preds = torch.argmax(logits, dim=1).tolist()
        
        # Almacenar las predicciones y las etiquetas verdaderas
        predictions.extend(preds)
        true_labels.extend(labels.tolist())

# Mostrar un reporte de clasificación
print(classification_report(true_labels, predictions, target_names=category_to_index.keys()))



