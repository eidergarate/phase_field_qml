# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:49:28 2024

@author: egarate
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_and_pandas(X_train, y_train, X_test, y_test): 
    
    scaler_x = StandardScaler()
    scaler_x.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler_x.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler_x.transform(X_test), columns=X_test.columns)

    # Escalar y_train y y_test con MinMaxScaler, configurando el rango (-1, 1)
    scaler_y_train = MinMaxScaler(feature_range=(-1, 1)).fit(y_train.values.reshape(-1, 1))
    y_train_scaled = pd.DataFrame(scaler_y_train.transform(y_train.values.reshape(-1, 1)))
    
    scaler_y_test = MinMaxScaler(feature_range=(-1, 1)).fit(y_test.values.reshape(-1, 1))
    y_test_scaled = pd.DataFrame(scaler_y_test.transform(y_test.values.reshape(-1, 1)))
    
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_y_train, scaler_y_test