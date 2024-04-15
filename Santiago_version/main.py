import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from train_step import train_step
from tester import TEST
from parameters import Params
from get_samples import get_samples
from Network import NeuralNetwork
from car_dynamics import Car_dyn
from loss_fn import loss_fn
from plot_trayectory import traj_plot



'''
start = torch.cat((possible_points[1,:,0],torch.tensor([Params['starting_speed'],Params['starting_orientation']])))

P2P_train(Params['epochs'],
          Params['car_params'],
          Params['Traj_lenght'],
          Params['Learning_rate'],
          start,
          possible_points[0,:,0]
          )
'''




def main(Params):
    # Set seed for repeatability
    #np.random.seed(0)
    #torch.manual_seed(0)


    # Check if the GPU is available
    device = torch.device("cpu") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Training device: {device}")

    system = Car_dyn(Params['A'].to(device), Params['B'].to(device))

    possible_points= get_samples(
    Params['batchs'],
    Params['points_per_batch'],
    Params['radius'],
    Params['Environment_limits'])

    x0 = torch.transpose(possible_points[0,:,0].unsqueeze(0),0,1).to(device)
    a=torch.randint( Params['points_per_batch'] , (1,Params['batchs'])) #this two arrays are just some indexing shuffleing to separete themn the test and train data.
    b=torch.randperm(Params['batchs'])

    train_batchs=torch.transpose(possible_points[b.squeeze(0)[0:int(0.8*Params['batchs']-1)],:,a.squeeze(0)[:int(0.8*Params['batchs']-1)]].unsqueeze(0),0,1).squeeze(1)
    test_batchs=torch.transpose(possible_points[b.squeeze(0)[int(0.8*Params['batchs']-1):Params['batchs']],:,a.squeeze(0)[int(0.8*Params['batchs']-1):Params['batchs']]].unsqueeze(0),0,1).squeeze(1)

    xf = torch.transpose(possible_points[1,:,0].unsqueeze(0),0,1).to(device)

    # Define the loss function
    criterion=loss_fn()

    #### TRAINING LOOP
    Controller = NeuralNetwork(4, 64, 64, 2)


    # Define the optimizer
    optimizer = optim.Adam(Controller.parameters(), lr= Params['Learning_rate'])

    Controller.to(device)
    train_loss_log = []


    for epoch in range(Params['epochs']):

        print('#################')
        print(f'# EPOCH {epoch}')
        print('#################')
        t_loss , f_traj = train_step(Controller, system, train_batchs, criterion, optimizer, device, xf, Params['Length'])
        #traj_plot(f_traj,xf)
        train_loss_log.append(t_loss)

    # Test the NN controller

    traj, _ = TEST(Controller, system, test_batchs[0].unsqueeze(1), Params['Length'], xf)
    traj_plot(traj.cpu(),xf.cpu())
   

if __name__ == "__main__":
    main(Params)
