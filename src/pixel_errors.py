# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:56:26 2024

@author: egarate
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_errors_pixels(y_real, y_pred):
    
    e_abs = np.abs(y_real - y_pred)
    
    return e_abs

def plot_errors_pixels(X, y_real, y_pred):
    
    e_abs = get_errors_pixels(y_real, y_pred)
    
    # Crear el gráfico de dispersión
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X['x'], X['y'], c=e_abs, cmap='viridis', s=100)
    # Añadir título y etiquetas
    
    plt.title('Absolute error real-predicted')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(scatter, ax=ax, label='OP_4000_error')

    return fig    