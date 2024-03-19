import numpy as np
import torch
from torch.optim import Adam
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
optimizer = Adam(model.parameters(), lr=0.1)
criterion = TrajectoryLoss()

input_sample = torch.tensor([x_start, y_start, x_end, y_end, speed_start, angle_start])

print(f"The input tensor is the following: ")
print(input_sample)

for i in range(5):

    # starting with initial position and speed
    x = x_start
    y = y_start
    speed = speed_start
    angle = angle_start

    # lists to store the trajectories for plotting
    #x_trajectory = torch.zeros(20)
    #y_trajectory = torch.zeros(20)

    # Perform trajectory
    for epoch in range(TrajectoryLength):
        # Call neural network to get desired speed and angle changes
        delta_speed, delta_angle = model(input_sample)

        # Update object's position
        obj = ObjectMovement(x, y, speed, angle)
        x, y, speed, angle = obj.move_object(delta_speed.item(), delta_angle.item())

        # Update the input sample
        input_sample = torch.tensor([x, y, x_end, y_end, speed, angle])

        # Keep the positions for plotting
        if epoch==0 : 
            x_trajectory=torch.tensor([x], requires_grad = True)
            y_trajectory=torch.tensor([y], requires_grad = True)
        else:
            x_trajectory=torch.cat((x_trajectory,x),0)
            y_trajectory=torch.cat((y_trajectory,y),0)
        #x_trajectory[epoch] = x
        #y_trajectory[epoch] = y
    
    x_end = torch.tensor(x_end, requires_grad=True)
    y_end = torch.tensor(y_end, requires_grad=True)
    
    loss = criterion(x_trajectory, y_trajectory, x_end, y_end)

    # Perform gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Total Loss:", loss)

    # Plot the trajectory
    plt.plot(x_trajectory.detach().numpy(), y_trajectory.detach().numpy(), marker='o')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Trajectory of the Object')
    plt.grid(True)
    plt.show()