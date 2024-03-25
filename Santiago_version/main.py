import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from point2point_trainner import P2P_train
from parameters import Params
from get_samples import get_samples

possible_points= get_samples(
    Params['batchs'],
    Params['points_per_batch'],
    Params['radius'],
    Params['Environment_limits'])

start = torch.cat((possible_points[1,:,0],torch.tensor([Params['starting_speed'],Params['starting_orientation']])))

P2P_train(Params['epochs'],
          Params['car_params'],
          Params['Traj_lenght'],
          Params['Learning_rate'],
          start,
          possible_points[0,:,0]
          )