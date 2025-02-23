import pandas as pd
import numpy as np

# Ruta del dataset original
INPUT_PATH = 'data/raw/old_data_carrers.csv'
# Se sobreescribe el mismo archivo (puedes cambiar OUTPUT_PATH si prefieres generar uno nuevo)
OUTPUT_PATH = 'data/raw/old_data_carrers_2.csv'

# Cargar el dataset (se supone que tiene dos columnas: "TEXTO" y "CARRERA")
df = pd.read_csv(INPUT_PATH)

total_records = len(df)
classes = sorted(df['CARRERA'].unique())
num_classes = len(classes)

# Definir la distribución balanceada:
# Se reparte el total de 1250 registros equitativamente. Si 1250 no es divisible exactamente
# se asigna 'base + 1' a las primeras 'remainder' clases.
base, remainder = divmod(total_records, num_classes)
target_counts = {}
for i, cl in enumerate(classes):
    target_counts[cl] = base + (1 if i < remainder else 0)

print("Distribución objetivo por clase:")
for cl in classes:
    print(f"  {cl}: {target_counts[cl]}")

# Procesar cada clase:
balanced_dfs = []
for cl in classes:
    group = df[df['CARRERA'] == cl]
    current_count = len(group)
    target = target_counts[cl]
    if current_count > target:
        # Si hay más registros de los necesarios, se muestrea sin reemplazo
        balanced_group = group.sample(n=target, random_state=42)
    elif current_count < target:
        # Si hay menos registros, se realiza oversampling (re-muestreo con reemplazo)
        additional_needed = target - current_count
        additional_samples = group.sample(n=additional_needed, replace=True, random_state=42)
        balanced_group = pd.concat([group, additional_samples])
    else:
        balanced_group = group
    balanced_dfs.append(balanced_group)

# Combinar todas las clases balanceadas
balanced_df = pd.concat(balanced_dfs)

# Para que el orden sea aleatorio (y evitar que queden agrupadas por clase)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Verificar el número final de registros (debe ser 1250)
final_count = len(balanced_df)
print(f"Total de registros después del balanceo: {final_count}")
if final_count != total_records:
    # Ajuste final si fuera necesario (esto es poco probable, pero lo incluimos para ser seguros)
    balanced_df = balanced_df.sample(n=total_records, random_state=42).reset_index(drop=True)
    print("Se ajustó el total de registros al número original.")

# Guardar el dataset balanceado (manteniendo la misma estructura de columnas)
balanced_df.to_csv(OUTPUT_PATH, index=False)
print(f"Archivo balanceado guardado en: {OUTPUT_PATH}") 