import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from car_dynamics import ObjectMovement
from P_controller import Prop_controller



def trajectory(model,criterion,controller_input,loss,Length,device):

  pos=torch.transpose(torch.transpose(controller_input,0,1)[0:2],0,1)
  kin=torch.transpose(torch.transpose(controller_input,0,1)[4:6],0,1)
  xf=torch.transpose(torch.transpose(controller_input,0,1)[2:4],0,1)
  obs = torch.transpose(torch.transpose(controller_input,0,1)[6:8],0,1)

  prop_controller = Prop_controller()

  for t in range(Length):
    out = model(controller_input)

    pd = prop_controller.forward(controller_input)

    system=ObjectMovement(pos,kin)

    pos,kin = system.dynamics(out,pd) 

    controller_input = torch.cat((pos,xf,kin,obs),1)

    loss = loss + criterion(pos,obs,xf)


  return loss