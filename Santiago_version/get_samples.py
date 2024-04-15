import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def get_samples(batchs=5,size=20,radius=1,limits = torch.tensor([[-10,10],[-10,10]])):
  centers= torch.rand(2,batchs)*limits.diff()+limits[:,0].unsqueeze(dim=1)
  angles=torch.rand(size)
  positions=centers.T.reshape(batchs,2,1) +  torch.rand(size)*torch.tensor([[radius],[radius]])*torch.cat((torch.cos(angles).unsqueeze(0),torch.sin(angles).unsqueeze(0)),0)
  #visualise data:
  for i in range(min(10,batchs)):
    plt.plot(positions[i,0,:],positions[i,1,:],'o',markersize=2)
  return positions

