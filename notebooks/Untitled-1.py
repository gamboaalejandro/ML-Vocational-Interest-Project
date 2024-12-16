# %%
DATASET_PATH = '../data/raw/data_carrers.csv'
OUTPUT_PATH = '../data/processed/processed-dataset.csv'
import pandas as pd

# Cargar el dataset
df = pd.read_csv(DATASET_PATH, encoding="UTF-8")

# Eliminar valores nulos y duplicados
df = df.dropna()
df = df.drop_duplicates()

# Preprocesar el texto
def preprocess_text(text):
    text = text.lower()
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    return text

df['TEXTO'] = df['TEXTO'].apply(preprocess_text)

# Mapear las categorías a índices numéricos
categories = df['CARRERA'].unique().tolist()
category_to_index = {category: idx for idx, category in enumerate(categories)}
df['LABEL'] = df['CARRERA'].map(category_to_index)

texts = df['TEXTO'].tolist()
labels = df['LABEL'].tolist()


# %%
import torch
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
from sklearn.model_selection import train_test_split
import accelerate
import transformers
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score




tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
labels = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_mask, labels)


# %%
from sklearn.model_selection import train_test_split

# Convertir tensores a numpy arrays
input_ids_np = input_ids.numpy()
attention_mask_np = attention_mask.numpy()
labels_np = labels.numpy()

# División en entrenamiento+validación y prueba
X_train_val, X_test, y_train_val, y_test, mask_train_val, mask_test = train_test_split(
    input_ids_np, labels_np, attention_mask_np, test_size=0.2, random_state=42, stratify=labels_np)

# División en entrenamiento y validación
X_train, X_val, y_train, y_val, mask_train, mask_val = train_test_split(
    X_train_val, y_train_val, mask_train_val, test_size=0.25, random_state=42, stratify=y_train_val)


from torch.utils.data import DataLoader
# Convertir a tensores
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(mask_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(mask_val), torch.tensor(y_val))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(mask_test), torch.tensor(y_test))

# Crear DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_dataloader = DataLoader(val_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)

from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertConfig

config = BertConfig.from_pretrained('bert-base-uncased', 
                                    num_labels=len(category_to_index), 
                                    hidden_dropout_prob=0.3, 
                                    attention_probs_dropout_prob=0.3)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)

optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)






# %%
## BUCLE DE ENTRENAMIENTO

epochs = 60
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
# Calcula los pesos de clase
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Define la función de pérdida con los pesos de clase
from torch.nn import CrossEntropyLoss
loss_fn = CrossEntropyLoss(weight=class_weights)


model.to(device)

model.train()

training_losses = []
validation_losses = []
validation_accuracies = []
from tqdm import tqdm
for epoch in range(epochs):
    ### Entrenamiento ###
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Entrenamiento"):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    training_losses.append(avg_train_loss)

    ### Validación ###
    model.eval()
    total_val_loss = 0
    val_predictions = []
    val_true_labels = []

    with torch.no_grad():
        for batch in tqdm(validation_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Validación"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            val_predictions.extend(preds.cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())
    
    avg_val_loss = total_val_loss / len(validation_dataloader)
    validation_losses.append(avg_val_loss)

    # Calcular precisión en validación
    val_accuracy = accuracy_score(val_true_labels, val_predictions)
    validation_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}/{epochs}, Pérdida Entrenamiento: {avg_train_loss:.4f}, Pérdida Validación: {avg_val_loss:.4f}, Precisión Validación: {val_accuracy:.4f}")

# %%
## MONITOREO DE GRÁFICOS Y METRICAS

import matplotlib.pyplot as plt

epochs_range = range(1, epochs+1)

# Graficar pérdidas
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, training_losses, label='Pérdida de Entrenamiento')
plt.plot(epochs_range, validation_losses, label='Pérdida de Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida por Época')

# Graficar precisión
plt.subplot(1, 2, 2)
plt.plot(epochs_range, validation_accuracies, label='Precisión de Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.title('Precisión por Época')

plt.show()



