import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from car_dynamics import Car_dyn
from plot_trayectory import traj_plot
from car_dynamics import ObjectMovement


def trajectory(model,criterion,state,kinematics,xf,loss,T):
  f_traj=state.detach().clone()
  for t in range(T):
            # Forward pass
    system=ObjectMovement(state,kinematics)
    out = model(torch.cat((state,kinematics,xf),1).transpose(0,1).reshape(1,-1))
    comand = torch.transpose(out,0,1)
    next_state,new_kinematics = system.dynamics(comand)
            # print(next_state, xf)
            # Compute loss
    loss = loss + criterion(next_state, xf)
    state = next_state
    kinematics = new_kinematics
    f_traj= torch.cat((f_traj,next_state.detach().clone()),1)
  return loss 
