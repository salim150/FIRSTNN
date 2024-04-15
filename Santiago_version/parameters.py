import torch

'''
Params= {
    'epochs':10,
    'car_params': torch.tensor([0.2,1,20]),
    'Traj_lenght':20,
    'Learning_rate':0.1,
    'batchs': 5,
    'points_per_batch':20,
    'radius':1,
    'Environment_limits':torch.tensor([[-10,10],[-10,10]]),
    'starting_speed':0,
    'starting_orientation':0
}
'''

Params = {
    'A': torch.tensor([[1, 0], [0, 1]], dtype=torch.float) ,
    'B': torch.tensor([[0.5, 0.3], [0.25, 0.4]], dtype=torch.float),
    'Length': 30,
    'batchs': 40,
    'points_per_batch':20,
    'radius':1,
    'Environment_limits':torch.tensor([[-10,10],[-10,10]]),
    'epochs': 100,
    'Learning_rate':1e-3,
    'max_speed':3,
    'max_omega':1,
    'max_acc':0.2,
    'max_ang_acc':0.1,   

    'alpha' : 1,
    'beta' : 2,
    'gamma' : 10,
    'xmin' : -10,
    'ymin' : -10,
    'xmax' : 10,
    'ymax' : 10,
    'obssize' : 3,
    'outside_penalty_value' : 200,  # Valeur de la pénalité pour sortir du terrain
    'obstacle_penalty_value' : 500 , # Valeur de la pénalité pour toucher un obstacle
    'high_value' : 100000,
}