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
    'Lenght': 50,
    'batchs': 5,
    'points_per_batch':20,
    'radius':1,
    'Environment_limits':torch.tensor([[-10,10],[-10,10]]),
    'epochs': 500,
    'Learning_rate':1e-2
    
}