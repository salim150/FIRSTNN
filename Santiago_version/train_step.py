import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from get_full_traj import trajectory
from Network import NeuralNetwork
from plot_trayectory import traj_plot
from loss_fn import loss_fn
from parameters import Params


def train_step(model,batched_ic,starting_kinematics ,criterion, optimizer, device, Length:int, printer=True):

    model.train()
    train_loss = []

    batched_ic


    dimm= Params['batchs size'] if (Params['batchs size']>1) else int(1)


    starting_kinematics=torch.kron(torch.ones(dimm,1).to(device),starting_kinematics).unsqueeze(1).to(device)


    # iterate over the batches
    for  i in range (Params['#of batchs']):
      if (i%100 == 0):print(i)

      loss=0          # initiate loss
      sample_batched = batched_ic[i*dimm:(i+1)*dimm]

      input_sample=torch.cat((sample_batched.reshape(dimm,1,4) , starting_kinematics),2).squeeze(1).to(device)
      # Perform trajectory
      loss = trajectory(model,criterion,input_sample,loss, Length,device)
      loss =loss/ Length


      # Perform gradient descent
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_loss.append(loss.cpu().detach().numpy())




    return train_loss


