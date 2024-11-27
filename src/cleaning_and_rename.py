# -*- coding: utf-8 -*-
"""
Created on 17.07.24

@author: mgomez
"""

import re 

def cleaning_and_rename(df):

    # Reemplazar los valores de los experimentos para que vayan de 1 a 7
    df['exp_id'].replace({64: 1, 65:2, 66: 3, 67:4, 68:5, 69:6, 70:7}, inplace=True)

    # Comprobar cuanto vale el parámetro theta_0 en cada exp_id
    df['theta0'].value_counts()

    # Borrar los otros parámetros
    df_clean = df.drop(['tau', 'alpha', 'epsilonbar', 'kappa', 'delta', 'j_parameter'], axis=1)

    # Insertar la última columna como la primera
    df_clean.insert(0, 'exp_id_new', df_clean['exp_id'])

    # Eliminar la columna 'unnamed:0' y la exp_id
    df_clean = df_clean.drop(['Unnamed: 0', 'exp_id'], axis=1)

    # Renombrar exp_id
    df_clean= df_clean.rename(columns={'exp_id_new': 'exp_id'})

    # RENOMBRAR
    nuevos_nombres = []

    for columna in df_clean.columns:
        nuevo_nombre = columna
        # Reemplazar nombres
        nuevo_nombre = re.sub(r'_pos', '', nuevo_nombre)
        nuevo_nombre = re.sub(r'up', 'u', nuevo_nombre)
        nuevo_nombre = re.sub(r'right', 'r', nuevo_nombre)
        nuevo_nombre = re.sub(r'left', 'l', nuevo_nombre)
        nuevo_nombre = re.sub(r'down', 'd', nuevo_nombre)
        # Reemplazar '-3900' por '100'
        nuevo_nombre = re.sub(r'-3000', '1000', nuevo_nombre)
        # Reemplazar '-1950' por '2500'
        nuevo_nombre = re.sub(r'-2000', '2000', nuevo_nombre)    
        # Reemplazar nombre del pixel actual
        nuevo_nombre = re.sub(r'\(xi, yi\)', '', nuevo_nombre)
        # Reemplazar nombre de la var de salida
        nuevo_nombre = re.sub(r'_0', '_4000', nuevo_nombre)
        # Reemplazar dos barras bajas
        nuevo_nombre = re.sub(r'__', '_', nuevo_nombre)
        # Agregar el nuevo nombre a la lista
        nuevos_nombres.append(nuevo_nombre)

    df_clean_renamed = df_clean

    # Renombrar las columnas con los nuevos nombres
    df_clean_renamed.rename(columns=dict(zip(df_clean.columns, nuevos_nombres)), inplace=True)

    # Ordenar el DataFrame por 'id', 'value1' y 'value2'
    df_clean_renamed = df_clean_renamed.sort_values(by=['exp_id', 'x', 'y'])

    # Reordenar el índice
    df_clean_renamed = df_clean_renamed.reset_index(drop=True)

    # Extraer el nombre de todas las columnas
    column_names = df_clean_renamed.columns.tolist()
    # Mover 'theta0' a la segunda posición
    column_names.insert(1, column_names.pop(column_names.index('theta0')))
    # Reordenar las columnas
    df_clean_renamed = df_clean_renamed[column_names]

    # Extraer los nombres de las columnas
    nombres_columnas = df_clean_renamed.columns.tolist()
    # Especificar la ruta del archivo de texto
    ruta_archivo = '../outputs/nombres_columnas.txt'
    # Escribir los nombres de las columnas en el archivo de texto
    with open(ruta_archivo, 'w') as archivo:
        for nombre_columna in nombres_columnas:
            archivo.write(nombre_columna + '\n')

    df_clean_renamed.to_csv('../data/df_clean_renamed.csv', index=False)

    return(df_clean_renamed)


def format_ids(ids):
    """
    Formatea los IDs para su escritura en un archivo de texto.
    
    Parameters:
        ids (int, list of int): Un entero o una lista de enteros.
        
    Returns:
        str: La representación en cadena de los IDs.
    """
    if isinstance(ids, list):
        return ", ".join(map(str, ids))
    else:
        return str(ids)
