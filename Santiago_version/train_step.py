import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from get_full_traj import trajectory
from Network import NeuralNetwork
from plot_trayectory import traj_plot
from loss_fn import loss_fn


'''
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
'''

def train_step(model, system, batched_ic, criterion, optimizer, device, xf:torch.Tensor, T:int, printer=True):

    model.train()
    train_loss = []
    loss = 0

    # iterate over the batches
    for sample_batched in batched_ic:

        # Move data to device
        state = sample_batched.to(device)

        # Iterate over the horizon [0,T]. At each iteration given the current state compute the control prediction of the NN. Then update the state with the model dynamics.
        # Finally compute the loss of current state wrt desired target

        loss_T , f_traj = trajectory(model,system,criterion,state,xf,loss,T)

        loss = loss_T/T
        # Backpropagation (Torch automatically computes the gradient)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # Save train loss for this batch

        train_loss.append(loss.detach().cpu().numpy())


    # Save average train loss over the batches
    train_loss = np.mean(train_loss)
    if(printer):
        print(f"AVERAGE TRAIN LOSS: {train_loss}")
        return train_loss , f_traj
