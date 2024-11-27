# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:50:45 2024

@author: egarate
"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

n_layers = 7
obs = "PauliX"


def qcircuit(weights, obs, n_layers, xi=None):

    """
    Parameters' dimensions must be:
    dim(weights) = (n_layers, n_qubits, 3)
    """
    
    n_qubits = 8

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


n_qubits = 8

dev = qml.device("default.qubit", wires = n_qubits)
qnode = qml.QNode(qcircuit, dev, interface = "torch")

xi = np.random.random(n_qubits) 

weights = np.array([[[r"\Theta_{}".format(i*3 + j + k) for k in range(3)] for j in range(n_qubits)] for i in range(n_layers)])
weights = np.array(weights, dtype=object).reshape(1, n_layers, n_qubits, 3)

# Llamada al circuito para dibujar
plt.rcParams.update({'font.size': 18})
fig, ax = qml.draw_mpl(qcircuit)(weights, obs, n_layers, xi)
plt.show()

# Guardar la imagen
fig.savefig("C:/Users/egarate/Desktop/kubit-qml/outputs/VQA_circuit.pdf", dpi=300)










