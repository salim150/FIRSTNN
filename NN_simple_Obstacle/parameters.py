import torch


Params = {
    'Network_layers': [6, 64,64,64, 2],
    'Length': 30,
    'batchs': 100,
    'points_per_batch':10,
    'radius':0.5,
    'Environment_limits':torch.tensor([[-10,10],[-10,10]]),
    'epochs': 50000,
    'Learning_rate':1e-3,
    'max_speed':3,
    'max_omega':1,
    'max_acc':0.2,
    'max_ang_acc':0.1,

    'car_size' : 0.6,
    'collision_safety' :0.1,
    'Trajectory_length':20,
    'start_radius' : 0,

    'alpha' : 1,
    'beta' : 2,
    'gamma' : 10,
    'obssize' : 3,
    'outside_penalty_value' : 10,  # Coefficient de la pénalité pour sortir du terrain
    'obstacle_penalty_value' : 10, # Coefficient de la pénalité pour toucher un obstacle
    'high_value' : 1000,
}