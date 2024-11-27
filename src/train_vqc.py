import pennylane as qml
import torch
import numpy as np 
import pandas as pd
import datetime
from predict import*
import torch
from torch.autograd import Variable
import torch.optim as optim


def qcircuit(weights, obs, n_layers, xi=None):

    """
    Parameters' dimensions must be:
    dim(weights) = (n_layers, n_qubits, 3)
    """
    
    n_qubits = len(xi)

    for l in range(n_layers):
        # Encoding
        for i, val in enumerate(xi):
            qml.RY(np.arctan(val) + np.pi/2, wires=i)  
        #Ansatz
        qml.StronglyEntanglingLayers(weights=weights[0][l,:,:].reshape(1,-1,3), wires=range(n_qubits))
        
    # Measurement
    if obs=="PauliX":
       m = qml.expval(qml.PauliX(0))
    elif obs=="PauliY":
       m = qml.expval(qml.PauliY(0))
    elif obs=="PauliZ":
       m = qml.expval(qml.PauliZ(0))
    return m


def evaluate_vqc(qcircuit, weights, n_layers, xi, obs):
# Obtain the result from the circuit measurement
  output = qcircuit(weights,obs, n_layers, xi)

  return output


def huber_loss(qcircuit, weights, n_layers, X, y, factor, obs, delta = 0.25):
    # Calculate Huber loss cost function

    loss_values = torch.tensor([], dtype=torch.float32)

    for i, xi in enumerate(X):    
    
        yi_pred = evaluate_vqc(qcircuit, weights, n_layers, xi, obs)*factor[0]+factor[1]
        
        error_i = y[i] - yi_pred
        
        if torch.abs(error_i) <= delta:
            
            loss_i = torch.square(error_i)/2
        else:
            
            loss_i = delta*(torch.abs(error_i) - delta/2)
        
        loss_values = torch.cat((loss_values, loss_i.unsqueeze(0)), dim=0)
    
    return torch.mean(loss_values)


def MSE_loss(circuit, weights, n_layers, X, y, factor, obs):
  # Calculate MSE loss cost function
  loss = 0
  n_samples = len(y)
  
  for i, xi in enumerate(X):
    yi_pred = evaluate_vqc(circuit, weights, n_layers, xi, obs)*factor[0]+factor[1]
    loss = loss + (yi_pred - y[i])**2
    
  #mean loss across all the observations  
  return loss/n_samples



def compute_loss(circuit, weights, n_layers, X, y, factor, obs, loss_type ):
    # Compute the value of the loss 
    if loss_type == "Huber":
    
        loss = huber_loss(circuit, weights, n_layers, X, y, factor, obs)

    elif loss_type == "MSE":
         
         loss = MSE_loss(circuit, weights, n_layers, X, y, factor, obs)
    else: 

        print ('Set Huber loss by default')
        loss = huber_loss(circuit, weights, n_layers, X, y, factor, obs)

    return loss

def initialize_parameters(n_layers, n_qubits, n_weights):

    #Initialize the weights and factor

    weights = [Variable(torch.randn(n_layers, n_qubits, n_weights), requires_grad=True)]
    factor = torch.tensor([1.0,0], requires_grad=True)

    return weights, factor

def initialize_optimizer(weights, factor): 

    #Initialize the Adam optimizer from torch. Define as variables weights and factor.

    optimizer = optim.Adam(weights, lr = 0.01)
    optimizer_factor = optim.Adam([factor], lr=0.01) 

    return optimizer, optimizer_factor
    
    

def train_vqc(n_layers, X_train_tensor, y_train_tensor, num_iters, batch_size, obs, loss_type): 
    """Training function with hyperparameters"""

    #Initialize the regression circuit
    n_qubits = X_train_tensor.shape[1]

    dev = qml.device("default.qubit", wires = n_qubits)
    qnode = qml.QNode(qcircuit, dev, interface = "torch")


    #Weights defined needs to be equal to the variational parameters in the Ansatz
    n_weights= 3
    weights, factor = initialize_parameters(n_layers, n_qubits, n_weights)
 

    #Initialize optimizers
    
    optimizer, optimizer_factor= initialize_optimizer (weights, factor)
    
    #Initialize results vectors
    costs = []
    factor_costs = []

    initial_time_execution = datetime.datetime.now()
    
    for i in range(num_iters):

        #Training of mini-batches
        batch_indices = np.random.choice(len(X_train_tensor), size = (batch_size,), replace = False)
        X_batch = X_train_tensor[batch_indices]
        y_batch = y_train_tensor[batch_indices]

        optimizer.zero_grad()
        current_cost = compute_loss(qnode, weights, n_layers, X_batch, y_batch, factor, obs, loss_type)
        current_cost.backward()
        optimizer.step()
        costs.append(current_cost.item())

        optimizer_factor.zero_grad()
        current_cost_f = compute_loss(qnode, weights, n_layers, X_batch, y_batch, factor, obs, loss_type)
        current_cost_f.backward()
        optimizer_factor.step()        
        factor_costs.append(current_cost_f.item())
    
        #print(f"iter {i}, cost: {current_cost}, total time: {datetime.datetime.now()- initial_time_execution}")
        
    
    return qnode, weights,factor, initial_time_execution
   
   