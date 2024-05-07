import torch
import numpy as np


Params = {
    'Network_layers': [8, 64,64,64, 2],
    'Length': 30,
    '#of points': 1000,
    'points_per_cloud':10,
    'radius':0.5,
    'Environment_limits':torch.tensor([[-10,10],[-10,10]]),
    'epochs': 10,
    'Learning_rate':1e-3,
    'max_speed':3,
    'max_omega':1,
    'max_acc':0.2,
    'max_ang_acc': np.radians(20),

      
    'start_radius' : 0.2,

    'Prop_coeff_distance' : 0.5,
    'Prop_coeff_angle' : 0.5,
    'Prop_coeff_speed' : 0.5,
    'nn_coeff_speed' : 6,
    'nn_coeff_angle' : 8,


    '#of batchs': 32*256,
    'batchs size': 1,
  
    'alpha' : 1,
    'beta' : 2,
    'gamma' : 3,
    'obssize' : 3,
    'outside_penalty_value' : 10,  # Coefficient de la pénalité pour sortir du terrain
    'obstacle_penalty_value' : 10, # Coefficient de la pénalité pour toucher un obstacle
    'high_value' : 100,
}
