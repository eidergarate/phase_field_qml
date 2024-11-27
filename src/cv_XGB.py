# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:41:49 2024

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
from scale_and_pandas import *
from train_xgb import *




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

def train_xgb_cv(df_clean_imputed, input_cols, output_col, x_min = 0, 
                 x_max = 50, y_min = 0, y_max = 50, n_cv = 10, id0 = 0, idf = 6, 
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

        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_y_train, scaler_y_test = scale_and_pandas(X_train_ml, y_train, X_test_ml, y_test)
        
        xgb_model = train_xgb(X_train_scaled, y_train_scaled)
        
        y_train_pred = xgb_model.predict(X_train_scaled)
        r2_train = r2_score(y_true = y_train_scaled, y_pred = y_train_pred)
        
        y_test_pred = xgb_model.predict(X_test_scaled)
        r2_test = r2_score(y_true = y_test_scaled, y_pred = y_test_pred)
        
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        
        X_train_scaled = pd.concat([X_train_scaled, X_train[['x','y']]], axis = 1)
        X_test_scaled = pd.concat([X_test_scaled, X_test[['x','y']]], axis = 1)
        
        y_test_pred_rescaled = scaler_y_test.inverse_transform(y_test_pred.reshape(-1, 1))
        y_train_pred_rescaled = scaler_y_train.inverse_transform(y_train_pred.reshape(-1, 1))
        y_test_rescaled = scaler_y_test.inverse_transform(y_test_scaled.values.reshape(-1, 1))
        
        save_results_xgb(id_train, id_test, r2_train, r2_test, X_test_scaled, y_test_rescaled, y_test_pred_rescaled)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        