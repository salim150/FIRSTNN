import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from car_dynamics import Car_dyn

'''
def trajectory(model,car_params,Lenght=20,start_parameters=torch.tensor([0,0,0,0],dtype=torch.float32),target=torch.tensor([[2],[2]]),dtype=torch.float32):

  trajectory= torch.tensor([[start_parameters[0]],[start_parameters[1]]]) #where the trajectory starts from, at the end it shape should be (2,Lenght)
  state = torch.cat((start_parameters,target.reshape(2,1).squeeze())) #the state of the car it will be updateted every at every step (Lenght times)
  car   = ObjectMovement(state[0:4])

  for i in range(Lenght):

    controller_output = model(state).clone() # The neural networks gives my the command for the car

    newstate= car.move_object(controller_output,car_params).clone()

    state[0:4]= newstate.clone() # going through the car dynamics to determine the state of the car after applying the controller output

    new_trajectory= torch.cat((trajectory,state[0:2].reshape(2,1)),1)

    trajectory = new_trajectory.clone()#.detach().requires_grad_(True) # add to show how all works but backward propagation

  return trajectory

'''

def trajectory(model,system,criterion,state,xf,loss,T):
  f_traj=state.detach().clone()
  for t in range(T):
            # Forward pass
    out = model(torch.transpose(state, 0, 1))
    next_state = system.dynamics(state, out)
            # print(next_state, xf)
            # Compute loss
    loss += criterion(next_state, xf)
    state = next_state
    f_traj= torch.cat((f_traj,next_state.detach().clone()),1)
  return loss , f_traj
