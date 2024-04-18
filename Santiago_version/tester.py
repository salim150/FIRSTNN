import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from car_dynamics import ObjectMovement

def TEST(model,controller_input,Length) -> list:
  model.eval()
  pos=controller_input[0:2]
  kin=controller_input[4:6]
  xf=controller_input[2:4]
  x_trajectory=controller_input[0].unsqueeze(0)
  y_trajectory=controller_input[1].unsqueeze(0)
  u = []
  for t in range(Length):

    out = model(controller_input)
    system=ObjectMovement(pos,kin)
    pos,kin = system.dynamics(out)
    controller_input= torch.tensor([pos[0], pos[1], xf[0], xf[1], kin[0], kin[1]])

    x_trajectory=torch.cat((x_trajectory,pos[0].unsqueeze(0)),0)
    y_trajectory=torch.cat((y_trajectory,pos[1].unsqueeze(0)),0)
    u.append(out.cpu().detach().numpy())

  f_traj = torch.stack((x_trajectory,y_trajectory),0)

  return f_traj,u

