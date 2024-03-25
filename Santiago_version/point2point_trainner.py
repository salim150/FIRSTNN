import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from get_full_traj import trajectory
from Network import NeuralNetwork
from plot_trayectory import tray_plot
from loss_fn import loss_fn



def P2P_train(epochs,car_params,Lenght=20,LR=0.1,start_parameters=torch.tensor([0,0,0,0],dtype=torch.float32),target = torch.tensor(([[2],[2]]),dtype=torch.float32)):
  model = NeuralNetwork()
  optimizer = optim.SGD(model.parameters(), lr=LR)
  criterion = loss_fn()

  for epoch in range(epochs):

    optimizer.zero_grad()

    trajector = trajectory(model,car_params,Lenght,start_parameters,target)

    tray_plot(trajector,target.squeeze())

    loss= criterion(trajector.T.reshape(Lenght+1,2),target*(torch.ones_like(trajector).T.reshape(Lenght+1,2)))

    torch.autograd.set_detect_anomaly(True)

    loss.backward()

    optimizer.step()

    print(f"Loss at epoch #{epoch}:", loss)

