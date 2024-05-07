import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from parameters import Params



def traj_plot(trajectory,obstacle,target,i):
    fig=plt.figure(i+1)
    # Plot arena square
    x_min, x_max = Params["Environment_limits"][0]
    y_min, y_max = Params["Environment_limits"][1]
    plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], 'k')
    plt.plot(trajectory[0,:].detach().numpy(),trajectory[1,:].detach().numpy(), marker='o')
    plt.plot(target[0].clone().detach(),target[1].clone().detach(),'r' ,marker='*')
    plt.plot(trajectory[0,0].clone().detach(),trajectory[1,0].clone().detach(),'b' ,marker='x')

    circle = plt.Circle((obstacle[0].detach().numpy(), obstacle[1].detach().numpy()), Params['obssize'], color='r', fill=False)
    plt.gca().add_patch(circle)
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Trajectory of the Object')
    plt.grid(True)
    plt.show()
    return