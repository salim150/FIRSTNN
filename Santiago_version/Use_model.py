import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from train_step import train_step
from tester import TEST
from parameters import Params
from get_samples import get_samples
from Network import NeuralNetwork
from car_dynamics import Car_dyn
from loss_fn import loss_fn
from plot_trayectory import traj_plot
from car_dynamics import ObjectMovement


def using_model(Params):
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # 2. Create model save path
    MODEL_NAME = "Umbumping_cars_V1.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


    device = torch.device("cpu") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Training device: {device}")

    # Instantiate a fresh instance of LinearRegressionModelV2
    loaded_model_1 = NeuralNetwork(Params['Network_layers'])


    # Load model state dict 
    loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # Put model to target device (if your data is on GPU, model will have to be on GPU to make predictions)
    loaded_model_1.to(device)



    print(f"Loaded model:\n{loaded_model_1}")
    print(f"Model on device:\n{next(loaded_model_1.parameters()).device}")


    possible_points= get_samples(
       Params['batchs'],
       Params['points_per_batch'],
       Params['radius'],
       Params['Environment_limits'])
    #organize the points into starting , ending points and divide them into test & train.
    x0 = torch.transpose(possible_points[0,:,0].unsqueeze(0),0,1)
    a=torch.randint( Params['points_per_batch'] , (1,Params['batchs'])) #this two arrays are just some indexing shuffleing to separete themn the test and train data.
    b=torch.randperm(Params['batchs'])
    train_batchs=torch.transpose(possible_points[b.squeeze(0)[0:int(0.8*Params['batchs']-1)],:,a.squeeze(0)[:int(0.8*Params['batchs']-1)]].unsqueeze(0),0,1).squeeze(1)
    test_batchs=torch.transpose(possible_points[b.squeeze(0)[int(0.8*Params['batchs']-1):Params['batchs']],:,a.squeeze(0)[int(0.8*Params['batchs']-1):Params['batchs']]].unsqueeze(0),0,1).squeeze(1)
    
    xf = torch.transpose(possible_points[1,:,0].unsqueeze(0),0,1)
    
    starting_kinematics= torch.tensor([[0],[0]])
    
    for epoch in range (10):
        x0=test_batchs[epoch].unsqueeze(1)
        input_sample = torch.tensor([x0[0], x0[1], xf[0], xf[1], starting_kinematics[0], starting_kinematics[1]])
    
        f_traj,_ =TEST(loaded_model_1,input_sample,Params['Length'])
        traj_plot(f_traj,xf,epoch)



if __name__ == "__main__":
    using_model(Params)

    