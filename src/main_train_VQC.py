# -*- coding: utf-8 -*-
"""
Created on 17.07.24

@author: mgomez
"""

import os 
import pandas as pd

# cambiar directorio del proyecto 
os.chdir('C:/Users/egarate/Desktop/kubit-qml/src')


from cleaning_and_rename import *
from train_test_split import *
from edge_pixels_inputation import *
from scale_and_tensor import *
from train_vqc import * 
from predict import *
from cv_VQC import *

df_raw = pd.read_csv('../data/data_t_1000_paper.csv')


# Clean raw data and rename variables 
df_clean = cleaning_and_rename(df_raw)

#imput NAs on cleaned data
df_clean_imputed = imput_df_nearest(df_clean)

n_train = 6
n_test = 1

# Set image zone
x_min = 0
x_max = 50
y_min = 0 
y_max = 50

# Set variables to be used as inputs and output 
input_cols = ['x', 'y', 'OP_2000', 'OP_r1_1000', 'T_2000', 'OP_1000', 'T_1000', 'T_d1_2000', 'T_l2_2000', 'OP_l1_2000' ]
output_col = 'OP_4000'


# Train VQC
n_layers = 7
num_iters = 100
batch_size = 16
obs = "PauliX"
loss_type = "Huber"

n_cv = 7
id0 = 1
idf = 7

train_vqc_cv(df_clean_imputed, input_cols, output_col, n_layers, num_iters, batch_size, obs, loss_type, x_min, 
                 x_max, y_min, y_max, n_cv, id0, idf, n_train, n_test, combs_predefined = True)

