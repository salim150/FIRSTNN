import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from car_dynamics import ObjectMovement

def TEST(model, kinematics,x0: torch.Tensor, T:int, xf: torch.Tensor) -> list:
    state = x0
    u = []
    f_traj=state.detach().clone()
    system= ObjectMovement(state,kinematics)
    with torch.no_grad():
        for t in range(T):
            out = model(torch.cat((state,kinematics,xf),1).transpose(0,1).reshape(1,-1))
            comand = torch.transpose(out,0,1)
            next_state,new_kinematics = system.dynamics(comand)
            state = next_state
            kinematics = new_kinematics
            f_traj= torch.cat((f_traj,next_state.detach().clone()),1)
            u.append(out.cpu().detach().numpy())
    return f_traj , u

