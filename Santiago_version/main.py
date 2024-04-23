import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from train_step import train_step
from tester import TEST
from parameters import Params
from get_samples import get_samples
from Network import NeuralNetwork
from loss_fn import loss_fn
from plot_trayectory import traj_plot


def main(Params):
  # Set seed for repeatability
  #np.random.seed(0)
  #torch.manual_seed(0)
  # 1. Create models directory
  MODEL_PATH = Path("models")
  MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # 2. Create model save path
  MODEL_NAME = "Umbumping_cars_V3.pth"
  MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


  device = torch.device("cpu") if torch.backends.mps.is_available() else torch.device("cpu")
  print(f"Training device: {device}")

  #Input sample

  possible_points= get_samples(
      Params['batchs'],
      Params['points_per_batch'],
      Params['radius'],
      Params['Environment_limits'])


  #organize the points into starting , ending points and divide them into test & train.
  #x0 = torch.transpose(possible_points[0,:,0].unsqueeze(0),0,1)
  a=torch.randint( Params['points_per_batch'] , (1,Params['batchs'])) #this two arrays are just some indexing shuffleing to separete themn the test and train data.
  b=torch.randperm(Params['batchs'])

  train_batchs_i=torch.transpose(possible_points[b.squeeze(0)[0:int(0.8*Params['batchs']-1)],:,a.squeeze(0)[:int(0.8*Params['batchs']-1)]].unsqueeze(0),0,1).squeeze(1)
  test_batchs_i=torch.transpose(possible_points[b.squeeze(0)[int(0.8*Params['batchs']-1):Params['batchs']],:,a.squeeze(0)[int(0.8*Params['batchs']-1):Params['batchs']]].unsqueeze(0),0,1).squeeze(1)

  #xf = torch.tensor([[0],[0]])

  a=torch.randint( Params['points_per_batch'] , (1,Params['batchs'])) #this two arrays are just some indexing shuffleing to separete themn the test and train data.
  b=torch.randperm(Params['batchs'])

  train_batchs_f=torch.transpose(possible_points[b.squeeze(0)[0:int(0.8*Params['batchs']-1)],:,a.squeeze(0)[:int(0.8*Params['batchs']-1)]].unsqueeze(0),0,1).squeeze(1)
  test_batchs_f=torch.transpose(possible_points[b.squeeze(0)[int(0.8*Params['batchs']-1):Params['batchs']],:,a.squeeze(0)[int(0.8*Params['batchs']-1):Params['batchs']]].unsqueeze(0),0,1).squeeze(1)

  train_batchs = torch.stack((train_batchs_i, train_batchs_f),1)
  test_batchs= torch.stack((test_batchs_i, test_batchs_f),1)


  starting_kinematics= torch.tensor([[0],[0]])



  # Create neural network model
  model = NeuralNetwork(Params['Network_layers'])

  # Define optimizer
  optimizer = optim.Adam(model.parameters(), lr=Params['Learning_rate'])
  criterion = loss_fn()
  model.to(device)
  train_loss_log = []
  for epoch in range(Params['epochs']):
        t_loss, _ = train_step(model,train_batchs,starting_kinematics ,criterion, optimizer, device, Params['Length'])
        if (epoch%1 == 0) :
           # Plot the trajectory
           print('#################')
           print(f'# EPOCH {epoch}')
           print('#################')
           print("Total Loss:", t_loss.detach().clone().numpy())
           '''
           x0=test_batchs[epoch//100][0].unsqueeze(1)
           xf=test_batchs[epoch//100][1].unsqueeze(1)
           input_sample = torch.tensor([x0[0], x0[1], xf[0], xf[1], starting_kinematics[0], starting_kinematics[1]])
           
           f_traj,_ =TEST(model,input_sample,Params['Length'])
           traj_plot(f_traj,xf,epoch//100)
           '''
        
        train_loss_log.append(t_loss)
    # 3. Save the model state dict
  print(f"Saving model to: {MODEL_SAVE_PATH}")
  torch.save(obj= model.state_dict(), f=MODEL_SAVE_PATH)

 
if __name__ == "__main__":
    main(Params)






