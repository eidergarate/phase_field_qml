# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:50:51 2024

@author: egarate
"""

import pandas as pd
import numpy as np
import random
from itertools import combinations
from output_generation import *
from cleaning_and_rename import *
from train_test_split import *
from edge_pixels_inputation import *
from scale_and_tensor import *
from train_vqc import * 
from predict import *



def get_cv_sets(n_cv = 10, id0 = 1, idf = 7, n_train = 2, n_test = 1):
    
    all_combinations = []
    identifiers = [i for i in range(id0, idf + 1)]
    

    for test_combination in combinations(identifiers, n_test):
        test_set = set(test_combination)

        remaining_identifiers = [x for x in identifiers if x not in test_set]
        for train_combination in combinations(remaining_identifiers, n_train):
            all_combinations.append((test_combination, train_combination))
    
    combs_cv = random.sample(all_combinations, n_cv)
    
    return combs_cv

def train_vqc_cv(df_clean_imputed, input_cols, output_col, n_layers, num_iters, batch_size, obs, loss_type, x_min = 0, 
                 x_max = 25, y_min = 0, y_max = 50, n_cv = 10, id0 = 0, idf = 6, 
                 n_train = 2, n_test = 1, combs_predefined = False):
    
    if(combs_predefined == False):
        combs_cv = get_cv_sets(n_cv, id0, idf, n_train, n_test)
    
    else:
        combs_cv = [((4,), (1, 5)),
                    ((7,), (1, 3)),
                    ((7,), (5, 6)),
                    ((7,), (4, 5)),
                    ((1,), (4, 5)),
                    ((2,), (6, 7)),
                    ((5,), (3, 4)),
                    ((4,), (5, 6)),
                    ((7,), (3, 6)),
                    ((6,), (2, 7)),
                    ((4,), (3, 5)),
                    ((6,), (1, 2)),
                    ((3,), (4, 5)),
                    ((1,), (3, 7)),
                    ((2,), (1, 3)),
                    ((2,), (1, 5)),
                    ((5,), (3, 6)),
                    ((3,), (2, 6)),
                    ((6,), (1, 3)),
                    ((6,), (3, 5))]
        
    
    for cv_iter in combs_cv:
        
        id_train = list(cv_iter[1])
        id_test = list(cv_iter[0])
        
        X_train, y_train, X_test, y_test = train_test_split(df_clean_imputed, id_train, id_test, x_min, x_max, y_min, y_max, input_cols, output_col)
        
        X_train_ml = X_train.drop(columns=['x','y'])
        X_test_ml = X_test.drop(columns=['x','y'])

        # Scale the data and convert it to tensors
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_y_train, scaler_y_test = scale_and_tensor(X_train_ml, y_train, X_test_ml, y_test)

        qnode, weights, factor, initial_time_execution = train_vqc(n_layers, X_train_tensor, y_train_tensor, num_iters, batch_size, obs, loss_type)

        # Predict train set
        pred_train, y_train_tensor, r2_train, execution_time_train, weights_train, factor_train = predict_tensor(qnode, weights, n_layers, X_train_tensor, y_train_tensor, factor, obs, initial_time_execution)

        # Predict test set
        pred_test, y_test_tensor, r2_test, execution_time_test, weights_test, factor_test = predict_tensor(qnode, weights, n_layers, X_test_tensor, y_test_tensor, factor, obs, initial_time_execution)
        
        pred_test_rescaled = scaler_y_test.inverse_transform(np.array(pred_test).reshape(-1, 1))
        pred_train_rescaled = scaler_y_train.inverse_transform(np.array(pred_train).reshape(-1, 1))
        y_test_rescaled = scaler_y_test.inverse_transform(np.array(y_test_tensor).reshape(-1, 1))
        

        save_results_vqc(id_train, id_test, execution_time_test, r2_train, r2_test, X_test, y_test_rescaled, pred_test_rescaled)
        
        

