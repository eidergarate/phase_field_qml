# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:36:15 2024

@author: mgomez
"""

import matplotlib.pyplot as plt
import os
from cleaning_and_rename import *
import pandas as pd
import numpy as np
from pixel_errors import *

def complete_image(data_exp, y, pred = True ): 
    if pred: 
        data_exp['OP_4000_pred'] = y
        y_col = 'OP_4000_pred'
    else: 
        data_exp['OP_4000'] = y
        y_col = 'OP_4000'

    n = len(data_exp)
    new_rows = 10000
    column_names = ['x', 'y', y_col]
    dataset_new = pd.DataFrame(np.nan, index=range(new_rows), columns=column_names)

    # Define start and end indices for each quadrant
    start_idx = 0
    end_idx = n

    # Quadrant 1
    x1 =  100 - data_exp['y'].values
    y1 = data_exp['x'].values
    OP_4000 = data_exp[y_col].values
    
    dataset_new.loc[start_idx:end_idx-1, 'x'] = x1
    dataset_new.loc[start_idx:end_idx-1, 'y'] = y1
    dataset_new.loc[start_idx:end_idx-1, y_col] = OP_4000

    # Update indices for next quadrant
    start_idx = end_idx
    end_idx = start_idx + n

    # Quadrant 2
    x2 = 100 - data_exp['x'].values
    y2 = 100 - data_exp['y'].values

    dataset_new.loc[start_idx:end_idx-1, 'x'] = x2
    dataset_new.loc[start_idx:end_idx-1, 'y'] = y2
    dataset_new.loc[start_idx:end_idx-1, y_col] = OP_4000

    # Update indices for next quadrant
    start_idx = end_idx
    end_idx = start_idx + n

    # Quadrant 3
    x3 =  data_exp['x'].values
    y3 = data_exp['y'].values
   
    dataset_new.loc[start_idx:end_idx-1, 'x'] = x3
    dataset_new.loc[start_idx:end_idx-1, 'y'] = y3
    dataset_new.loc[start_idx:end_idx-1, y_col] = OP_4000

    # Update indices for next quadrant
    start_idx = end_idx
    end_idx = start_idx + n

    # Quadrant 4

    x4 =  data_exp['y'].values
    y4 = 100 -data_exp['x'].values
    
    dataset_new.loc[start_idx:end_idx-1, 'x'] = x4
    dataset_new.loc[start_idx:end_idx-1, 'y'] = y4  
    dataset_new.loc[start_idx:end_idx-1, y_col] = OP_4000

    return dataset_new


def plot_phasefield(X, y, pred):
    
    # Crear el gráfico de dispersión
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X['x'], X['y'], c=y, cmap='viridis', s=100, vmin=0, vmax=1)
    # Añadir título y etiquetas
    if pred:
        plt.title('Predicted Values of OP_4000')
    else: 
        plt.title('Actual Values of OP_4000')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(scatter, ax=ax, label='OP_4000')
    
    return fig

def save_results_vqc(id_train, id_test, execution_time_test, r2_train, r2_test,
                 X_test, y_test, pred_test):
    
    experiment_name = 'train' + str(id_train) + '_test' + str(id_test)
    output_dir = "/mnt/datastore-ia/KUBIT/outputs/" + experiment_name

    # Create folder if do not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Formatear el tiempo de ejecución
    execution_time_test_str = str(execution_time_test)

    # Save numerical results in txt 
    results_txt_path = os.path.join(output_dir, "results_vqc.txt")

    with open(results_txt_path, "w") as file:
            file.write("Train IDs:\n")
            file.write(format_ids(id_train) + "\n")
            file.write("Test IDs:\n")
            file.write(format_ids(id_test) + "\n")
            file.write(f"R^2 Score Train: {r2_train}\n\n")
            file.write(f"R^2 Score Test: {r2_test}\n\n")
            file.write(f"\nExecution Time Test: {execution_time_test_str} seconds\n")

    # Crear el gráfico usando la función plot_experiment
    plot_test = plot_phasefield(X_test, y_test, pred=False)
    # Guardar el gráfico
    plot_test.savefig(os.path.join(output_dir, "actual_test.jpg"))
    # Cerrar la figura
    plt.close(plot_test)

    # Crear el gráfico usando la función plot_experiment
    plot_pred_test = plot_phasefield(X_test, pred_test, pred=True)
    # Guardar el gráfico
    plot_pred_test.savefig(os.path.join(output_dir, "predicted_test_vqc.jpg"))
    # Cerrar la figura
    plt.close(plot_pred_test)

    # Crear grafico completo
    dataset_real = complete_image(X_test, y_test, pred=False )
    plot_test_all = plot_phasefield(dataset_real, dataset_real['OP_4000'], pred = False)
    # Guardar el gráfico
    plot_test_all.savefig(os.path.join(output_dir, "actual_test_complete.jpg"))
    # Cerrar la figura
    plt.close(plot_test_all)

    # Crear grafico completo
    dataset_pred = complete_image(X_test, pred_test, pred=True)
    plot_pred_all = plot_phasefield(dataset_pred, dataset_pred['OP_4000_pred'], pred = True)
    # Guardar el gráfico
    plot_pred_all.savefig(os.path.join(output_dir, "pred_test_complete.jpg"))
    plt.show(plot_pred_all)
    # Cerrar la figura
    plt.close(plot_pred_all)
    
    #Crear grafico de errores absolutos
    plot_abs_error = plot_errors_pixels(dataset_real, dataset_real['OP_4000'], dataset_pred['OP_4000_pred'])
    # Guardar el gráfico
    plot_abs_error.savefig(os.path.join(output_dir, "abs_error_complete.jpg"))
    # Cerrar la figura
    plt.close(plot_abs_error)
    
    
def save_results_xgb(id_train, id_test, r2_train, r2_test, X, y, y_pred):
    
    experiment_name = 'train' + str(id_train) + '_test' + str(id_test)
    output_dir = "//datastore.tekniker.es/ia/data-analytics/KUBIT/outputs/" + experiment_name
    # Create folder if do not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Save numerical results in txt 
    results_txt_path = os.path.join(output_dir, "results_xgb.txt")

    with open(results_txt_path, "w") as file:
            file.write("Train IDs:\n")
            file.write(format_ids(id_train) + "\n")
            file.write("Test IDs:\n")
            file.write(format_ids(id_test) + "\n")
            file.write(f"R^2 Score Train: {r2_train}\n\n")
            file.write(f"R^2 Score Test: {r2_test}\n\n")

    # Crear el gráfico usando la función plot_experiment
    plot_test = plot_phasefield(X, y, pred=False)
    # Guardar el gráfico
    plot_test.savefig(os.path.join(output_dir, "xgb_actual_test.jpg"))
    # Cerrar la figura
    plt.close(plot_test)
    plt.show(plot_test)

    # Crear el gráfico usando la función plot_experiment
    plot_pred_test = plot_phasefield(X, y_pred, pred=True)
    # Guardar el gráfico
    plot_pred_test.savefig(os.path.join(output_dir, "xgb_predicted_test_vqc.jpg"))
    # Cerrar la figura
    plt.close(plot_pred_test)

    # Crear grafico completo
    dataset_real = complete_image(X, y, pred=False )
    plot_test_all = plot_phasefield(dataset_real, dataset_real['OP_4000'], pred = False)
    # Guardar el gráfico
    plot_test_all.savefig(os.path.join(output_dir, "xgb_actual_test_complete.jpg"))
    # Cerrar la figura
    plt.close(plot_test_all)

    # Crear grafico completo
    dataset_pred = complete_image(X, y_pred, pred=True)
    plot_pred_all = plot_phasefield(dataset_pred, dataset_pred['OP_4000_pred'], pred = True)
    # Guardar el gráfico
    plot_pred_all.savefig(os.path.join(output_dir, "xgb_pred_test_complete.jpg"))
    # Cerrar la figura
    plt.close(plot_pred_all)
    
    #Crear grafico de errores absolutos
    plot_abs_error = plot_errors_pixels(dataset_real, dataset_real['OP_4000'], dataset_pred['OP_4000_pred'])
    # Guardar el gráfico
    plot_abs_error.savefig(os.path.join(output_dir, "xgb_abs_error_complete.jpg"))
    # Cerrar la figura
    plt.close(plot_abs_error)
    
    
    
    
    
    
    
    
    
    