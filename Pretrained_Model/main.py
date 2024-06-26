import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from train_step import train_step
from tester import TEST
from parameters import Params
from get_samples import get_samples
from Network import NeuralNetwork
from loss_fn import loss_fn2
from plot_trayectory import traj_plot



def main(Params):
  # Set seed for repeatability
  #np.random.seed(0)
  #torch.manual_seed(0)
  # 1. Create models directory
  MODEL_PATH = Path("models")
  MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # 2. Create model save path
  MODEL_NAME = "Umbumping_cars_V5.pth"
  MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


  device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
  print(f"Training device: {device}")

  #Input samples


  starting_kinematics= torch.tensor([0,0]).to(device)
  starting_kinematics_t=torch.kron(starting_kinematics,torch.ones(Params['batchs size'],1).to(device)).to(device)

  # Create neural network model
  model = NeuralNetwork(Params['Network_layers'],device)
  model.init_weights_zeros


  # Define optimizer
  optimizer = optim.Adam(model.parameters(), lr=Params['Learning_rate'])
  criterion = loss_fn2()
  train_loss = []


  for epoch in range(Params['epochs']):
     train_batchs , test_batchs, obstacle= get_samples(
         Params['obssize'],device,
         Params['#of points'],
         Params['car_size'],
         Params['Environment_limits'])
     obstacle_t= torch.kron(obstacle,torch.ones(Params['batchs size'],1)).to(device)

     
     t_loss = train_step(model,train_batchs,obstacle_t,starting_kinematics_t ,criterion, optimizer, device, Params['Length'])
     if (epoch%5 == 0) :
           # Plot the trajectory
           print('#################')
           print(f'# EPOCH {epoch}')
           print('#################')
           print("Total Loss:",np.mean(t_loss))
           #fig=plt.figure(1)
           #plt.plot(np.arange((0.8*Params['#of points'])/Params['batchs size']),t_loss)
           #plt.show()
           for i in range (0):
            x0=test_batchs[epoch//100+i][0]
            xf=test_batchs[epoch//100+i][1]
            input_sample =  torch.cat((x0,xf, starting_kinematics, obstacle.to(device)),0).unsqueeze(0).to(device)

            f_traj,_ =TEST(model,input_sample,Params['Length'],device)
            traj_plot(f_traj.cpu(),obstacle.cpu(),xf.cpu(),epoch//100+i)

     train_loss.append(t_loss)
    # 3. Save the model state dict
  print(f"Saving model to: {MODEL_SAVE_PATH}")
  torch.save(obj= model.state_dict(), f=MODEL_SAVE_PATH)
 
if __name__ == "__main__":
    main(Params)






