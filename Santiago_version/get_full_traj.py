import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from car_dynamics import Car_dyn
from plot_trayectory import traj_plot
from car_dynamics import ObjectMovement



def trajectory(model,criterion,controller_input,loss,Length,device):

  pos=controller_input[:,0:2]
  kin=controller_input[:,4:6]
  xf=controller_input[:,2:4]


  for t in range(Length):
    out = model(controller_input).to(device)

    system=ObjectMovement(pos,kin)

    pos,kin = system.dynamics(out)

    controller_input= torch.cat((pos,xf, kin),1)

    loss = loss + criterion(pos,xf)


  return loss 