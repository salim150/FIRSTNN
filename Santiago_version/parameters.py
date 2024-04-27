import torch
import numpy as np


Params = {
    'Network_layers': [6, 64,64,64, 2],
    'Length': 30,
    '#of points': 256*40+2,
    'points_per_batch':10,
    'radius':0.5,
    'Environment_limits':torch.tensor([[-10,10],[-10,10]]),
    'epochs': 1,
    'Learning_rate':1e-3,
    'max_speed':3,
    'max_omega':1,
    'max_acc':0.2,
    'max_ang_acc': np.radians(20),
  

    'alpha' : 1,
    'beta' : 2,
    'gamma' : 10,
    'obssize' : 3,
    'outside_penalty_value' : 200,  # Valeur de la pénalité pour sortir du terrain
    'obstacle_penalty_value' : 500 , # Valeur de la pénalité pour toucher un obstacle
    'high_value' : 100000,
}