import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from get_full_traj import trajectory
from Network import NeuralNetwork
from plot_trayectory import traj_plot
from loss_fn import loss_fn

def train_step(model,batched_ic,starting_kinematics ,criterion, optimizer, device, Length:int, printer=True):

    model.train()
    train_loss = []

    print(batched_ic.shape)
    # iterate over the batches
    for  sample_batched in batched_ic:
      print(sample_batched.shape)
      loss=0          # initiate loss
      input_sample = torch.tensor([sample_batched[0][0], sample_batched[0][1],
                                   sample_batched[1][0], sample_batched[1][1],
                                   starting_kinematics[0], starting_kinematics[1]]).to(device)

      # Perform trajectory
      loss,f_traj = trajectory(model,criterion,input_sample,loss, Length,device)
      loss =loss/ Length


      # Perform gradient descent
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_loss.append(loss.cpu().detach().numpy())




    return train_loss,f_traj