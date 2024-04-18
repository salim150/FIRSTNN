import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from network import create_nn
from dynamics import ObjectMovement
import matplotlib.pyplot as plt
from loss import TrajectoryLoss

#Setting the defined area of where the trajectory can be made
#The mac and min will be the defined interval of the x- and y-axis

max_value = 10
min_value = -10

#Input sample
x_start = torch.rand(1) * min_value
y_start = torch.rand(1) * min_value
x_end = torch.rand(1) * max_value
y_end = torch.rand(1) * max_value
speed_start = 0
angle_start = 0

TrajectoryLength = 20

# Create neural network model
model = create_nn()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.)
criterion = TrajectoryLoss()
loss_fn = nn.MSELoss()


input_sample = torch.tensor([x_start, y_start, x_end, y_end, speed_start, angle_start])

print(f"The input tensor is the following: ")
print(input_sample)

for epoch in range(10):

    optimizer.zero_grad()
    # starting with initial position and speed
    x_trajectory= torch.tensor([x_start],requires_grad=True)
    y_trajectory= torch.tensor([y_start],requires_grad=True)
    x=x_trajectory[0].detach()
    y=y_trajectory[0].detach()
    speed = speed_start
    angle = angle_start

    tarjet= torch.cat((x_end*torch.ones(21, requires_grad=True), y_end*torch.ones(21, requires_grad=True)),0)
    # lists to store the trajectories for plotting
    #x_trajectory = torch.zeros(20)
    #y_trajectory = torch.zeros(20)

    # Perform trajectory
    for i  in range(TrajectoryLength):
        # Call neural network to get desired speed and angle changes
        delta_speed, delta_angle = model(input_sample)

        # Update object's position
        obj = ObjectMovement(x, y, speed, angle)
        x, y, speed, angle = obj.move_object(delta_speed, delta_angle)

        # Update the input sample
        input_sample = torch.tensor([x, y, x_end, y_end, speed, angle])

        # Keep the positions for plotting
        x_trajectory = torch.cat((x_trajectory,torch.tensor([x],requires_grad=True)),0)
        y_trajectory = torch.cat((y_trajectory,torch.tensor([y],requires_grad=True)),0)

    

    
    loss =torch.tensor( loss_fn(torch.cat((x_trajectory, y_trajectory),0), tarjet),requires_grad=True)

    # Perform gradient descent
    loss.backward()

    optimizer.step()

    print("Total Loss:", loss)

    # Plot the trajectory
    plt.plot(x_trajectory.detach().numpy(), y_trajectory.detach().numpy(), marker='o')
    plt.plot(x_end.detach().numpy(),y_end.detach().numpy(), '.','r')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Trajectory of the Object')
    plt.grid(True)
    plt.show()