import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def tray_plot(trayectory,target):
    fig=plt.figure()
    plt.plot(trayectory[0,:].detach().numpy(),trayectory[1,:].detach().numpy(), marker='o')
    plt.plot(target[0].clone().detach(),target[1].clone().detach(), '.')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Trajectory of the Object')
    plt.grid(True)
    plt.show()
    return