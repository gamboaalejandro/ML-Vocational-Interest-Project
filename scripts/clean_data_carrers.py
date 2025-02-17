#!/usr/bin/env python3
"""
Script para limpiar el archivo data_carrers.csv eliminando las comas internas
del campo de texto, manteniendo intacto el separador que delimita el campo carrera.
"""

import argparse

def remove_internal_commas(line: str) -> str:
    """
    Toma una línea de CSV asumiendo dos columnas (TEXTO y CARRERA)
    y elimina todas las comas del primer campo (TEXTO).
    """
    # Utilizamos rsplit con maxsplit=1 para separar la última coma, que corresponde al delimitador de campo
    parts = line.rsplit(',', 1)
    if len(parts) < 2:
        # Si no se consigue dividir en dos campos, retorna la línea sin cambios
        return line
    text_field, career = parts
    # Eliminar todas las comas del texto
    text_field_clean = text_field.replace(',', '')
    return f"{text_field_clean},{career}"

def main(input_file: str, output_file: str) -> None:
    """
    Lee el archivo CSV de entrada, procesa línea por línea eliminando las comas
    internas del campo de texto y escribe el resultado en el archivo de salida.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        # Leer y escribir el encabezado (se asume que es 'TEXTO,CARRERA')
        header = infile.readline()
        outfile.write(header)
        
        # Procesar cada línea restante
        for line in infile:
            line = line.strip()
            if not line:
                continue  # omitir líneas vacías
            cleaned_line = remove_internal_commas(line)
            outfile.write(cleaned_line + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Elimina comas internas del campo de texto en data_carrers.csv"
    )
    parser.add_argument(
        "input_file", help="Ruta del archivo de entrada (por ejemplo: data/raw/data_carrers.csv)"
    )
    parser.add_argument(
        "output_file", help="Ruta del archivo de salida (por ejemplo: data/raw/data_carrers_clean.csv)"
    )
    args = parser.parse_args()

    main(args.input_file, args.output_file) 