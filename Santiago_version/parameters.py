import torch


Params= {
    'epochs':10,
    'Traj_lenght':20,
    'Learning_rate':0.1,
    'batchs': 5,
    'points_per_batch':20,
    'radius':1,
    'Environment_limits':torch.tensor([[-10,10],[-10,10]]),
    'starting_speed':0,
    'starting_orientation':0
}

