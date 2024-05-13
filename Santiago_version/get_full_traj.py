import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from car_dynamics import ObjectMovement
from P_controller import Prop_controller



def trajectory(model,criterion,controller_input,loss,Length,device):

  pos=controller_input[0:2]
  kin=controller_input[4:6]
  xf=controller_input[2:4]
  obs = controller_input[6:8]
  prop_controller = Prop_controller()

  for t in range(Length):
    out = model(controller_input).to(device)

    pd = prop_controller.forward(controller_input)

    system=ObjectMovement(pos,kin)

    pos,kin = system.dynamics(out,pd)

    controller_input= torch.cat((pos,xf, kin, obs),0)

    loss = loss + criterion(pos,obs,xf)


  return loss 