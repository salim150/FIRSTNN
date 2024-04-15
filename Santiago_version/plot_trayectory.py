import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from parameters import Params

def traj_plot(trayectory,target):
    fig=plt.figure()
    # Plot arena square
    x_min, x_max = Params["Environment_limits"][0]
    y_min, y_max = Params["Environment_limits"][1]
    plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], 'k')
    plt.plot(trayectory[0,:].detach().numpy(),trayectory[1,:].detach().numpy(), marker='o')
    plt.plot(target[0].clone().detach(),target[1].clone().detach(), '.')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Trajectory of the Object')
    plt.grid(True)
    plt.show()
    return