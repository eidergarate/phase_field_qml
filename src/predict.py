from train_vqc import *
import datetime

def evaluate_vqc(qcircuit, weights, n_layers, xi, obs):
  # Obtain the result from the circuit measurement
  output = qcircuit(weights,obs, n_layers, xi)
  return output

def quantum_predict(qircuit, weights, n_layers, X, factor, obs):
    #Predict the movel values
    y_pred = []
    for i, xi in enumerate(X):
        yi_pred = evaluate_vqc(qircuit, weights, n_layers, xi, obs)*factor[0]+factor[1]
        y_pred.append(yi_pred.item())
    return y_pred

def get_r2_score(y, y_pred):
    #Calculation of the R^2 metric
    residuals_squares = 0
    total_squares = 0
    mean = y.mean()

    for l, p in zip(y, y_pred):
        residuals_squares += (l - p)**2
        # total_squares +=l**2 #Esto es incorrecto, a no ser que mean(y)==0 
        total_squares += (l-mean)**2
        
    r2_score_value = 1 - residuals_squares/total_squares
    return r2_score_value


def predict_tensor(qnode, weights, n_layers, X_tensor,y_tensor, factor, obs, initial_time_execution):
    #Prediction of training set
    predictions = quantum_predict(qnode, weights, n_layers, X_tensor, factor, obs)
  
    #r^2 score calculation
    r2 = get_r2_score(y_tensor, predictions)

    # Calculation of execution time 
    final_time_execution = datetime.datetime.now()
    execution_time = final_time_execution- initial_time_execution
    
    return predictions, y_tensor, r2, execution_time, weights, factor