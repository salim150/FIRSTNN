import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from car_dynamics import ObjectMovement
from P_controller import Prop_controller

def TEST(model,controller_input,Length,device) -> list:
  model.eval()

  pos=controller_input[0:2]
  kin=controller_input[4:6]
  xf=controller_input[2:4]
  obs = controller_input[6:8]
  prop_controller = Prop_controller()

  x_trajectory=pos[0].unsqueeze(0)
  y_trajectory=pos[1].unsqueeze(0)
  u = []
  for t in range(Length):
    
    out = model(controller_input).to(device)
    pd = prop_controller.forward(controller_input)

    system=ObjectMovement(pos,kin)
    pos,kin = system.dynamics(out,pd)

    controller_input= torch.cat((pos,xf, kin, obs),0)

    x_trajectory=torch.cat((x_trajectory,pos[0].unsqueeze(0)),0)
    y_trajectory=torch.cat((y_trajectory,pos[1].unsqueeze(0)),0)
    u.append(out.cpu().detach().numpy())

  f_traj = torch.stack((x_trajectory,y_trajectory),0)

  return f_traj,u

