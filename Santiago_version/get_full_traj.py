import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from car_dynamics import Car_dyn



def trajectory(model,system,criterion,state,xf,loss,T):
  f_traj=state.detach().clone()
  for t in range(T):
            # Forward pass
    out = model(torch.cat((state,xf),1).transpose(0,1).reshape(1,-1))
    comand = torch.transpose(out,0,1)
    next_state = system.dynamics(state, comand)
            # print(next_state, xf)
            # Compute loss
    loss += criterion(next_state, xf)
    state = next_state
    f_traj= torch.cat((f_traj,next_state.detach().clone()),1)
  return loss , f_traj
