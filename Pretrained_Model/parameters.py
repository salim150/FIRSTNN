import torch
import numpy as np


Params = {
    'Network_layers': [8, 64,64,64, 2],
    'Length': 30,
    '#of points': 1000*8,
    'Environment_limits':torch.tensor([[-10,10],[-10,10]]),
    'epochs': 300,
    'Learning_rate':1e-4,
    'dt':0.2,


    'max_speed':torch.tensor(2),
    'max_omega':1,
    'max_acc':torch.tensor(2),
    'max_ang_acc': torch.tensor(torch.pi/4),


    'collision_safety' : 0.5,


    'start_radius' : 0.2,

    'car_size' : 1,

    'Prop_coeff_distance' : 0.5,
    'Prop_coeff_angle' : 0.5,
    'Prop_coeff_speed' : 0.5,
    'nn_coeff_speed' : 6,
    'nn_coeff_angle' : 8,


    'batchs size': 256,

    'alpha' : 8,
    'beta' : 50,
    'gamma' : 1,
    'obssize' : 1,
    'outside_penalty_value' : 10,  # Coefficient de la pénalité pour sortir du terrain
    'obstacle_penalty_value' : 10, # Coefficient de la pénalité pour toucher un obstacle
    'high_value' : 1000,
}