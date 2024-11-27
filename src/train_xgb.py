# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:03:50 2024

@author: egarate
"""

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def train_xgb(X_train, y_train):
    
    xgb_model = xgb.XGBRegressor()
    
    param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
    }

    grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='r2',  
    cv=5
    )

    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    return best_model

    
    