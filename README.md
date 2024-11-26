# phase_field_qml

# Description
Repository created to evaluate the viability of using a Variational Quantum Algorithms (VQAs) as surrogate models for the Phase Field problem in metal solidification patterns.

## Purpose
This repository trains and evaluates VQAs and is compared to a classical ML algorithm, Extreme Gradient-Boosting algorithm (XGBoost).

## Outcomes
Training creates different results stored in _Outputs_. 
There is a folder for each validation iteration indicating by the name training and testing experiment id-s. 
For each type of model 5 images are stored: absolute errors pixel-by-pixel, and partial and complete image actuals and predictions of the geometries.
For the model performance metrics plain text files are stored  with R2 scores and computational costs.

## Requirements
Project has been developed in a conda environment. Conda environment dependencies are stored in *environment.yml*.

```python
conda env create -f environment.yml
```
## System Version
Developed in a Windows 10 (64 bits).
Tested in a Windows 10 (64 bits) and a Ubuntu 22.04.3 LTS (64 bits).

**Local Windows PC hardware:**

Intel Core i5-9500 CPU processor of 6 cores \
8GB of RAM 

**Server's Ubuntu hardware**

Intel Xenon CPU processor of 32 cores \
128GB of RAM \
Nvidia Tesla T4 TU104GL

## Quantum Computing
This repository has been developed using *pennylane* Quantum Computing open source library. More information and demos are available in:

**Pennylane:**\
https://pennylane.ai/

# Data description

TODO

# Usage

TODO

# Authors and Contributors
**Eider Garate** \
Contact: eider.garate\@tekniker.es \

**Meritxell Gomez** \
Contact: meritxell.gomez\@tekniker.es \

**Jon Lambarri** \
Contact: jon.lambarri\@tekniker.es \

**Naroa Martin** \
Contact: naroa.martin@tekniker.es
Contact: jon.lambarri\@tekniker.es \





