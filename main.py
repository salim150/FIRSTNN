import numpy as np
import torch
from torch.optim import Adam
from network import create_nn
from dynamics import ObjectMovement
import matplotlib.pyplot as plt

#Setting the defined area of where the trajectory can be made
#The mac and min will be the defined interval of the x- and y-axis

max_value = 10
min_value = -10

#Input sample
x_start = np.random.uniform(low=min_value, high=0)
y_start = np.random.uniform(low=min_value, high=0)
x_end = np.random.uniform(low=0, high=max_value)
y_end = np.random.uniform(low=0, high=max_value)
speed_start = 0
angle_start = 0

TrajectoryLength = 20

# Create neural network model
model = create_nn()

# Define optimizer
optimizer = Adam(model.parameters(), lr=1)

input_sample = torch.tensor([x_start, y_start, x_end, y_end, speed_start, angle_start])

print(f"The input tensor is the following: ")
print(input_sample)

for i in range(10):

    # starting with initial position and speed
    x = x_start
    y = y_start
    speed = speed_start
    angle = angle_start

    # List to store losses at each step
    losses = torch.zeros(20)

    # lists to store the trajectories for plotting
    x_trajectory = torch.zeros(20)
    y_trajectory = torch.zeros(20)

    # Perform trajectory
    for epoch in range(TrajectoryLength):
        # Call neural network to get desired speed and angle changes
        delta_speed, delta_angle = model(input_sample)

        # Update object's position
        obj = ObjectMovement(x, y, speed, angle)
        x, y, speed, angle = obj.move_object(delta_speed.item(), delta_angle.item())

        # Update the input sample
        '''input_sample[0] = x
        input_sample[1] = y
        input_sample[4] = speed
        input_sample[5] = angle'''
        input_sample = torch.tensor([x, y, x_end, y_end, speed, angle])

        # Compute loss (distance between current position and goal)
        losses[epoch] = torch.sqrt((torch.tensor(x_end) - x) ** 2 + (torch.tensor(y_end) - y) ** 2)

        # Keep the positions for plotting
        x_trajectory[epoch] = x
        y_trajectory[epoch] = y

    # Accumulate losses
    total_loss = sum(losses)

    # Perform gradient descent
    optimizer.zero_grad()
    total_loss_tensor = torch.tensor(total_loss, dtype=torch.float32, requires_grad=True)
    total_loss_tensor.backward()
    optimizer.step()

    print("Total Loss:", total_loss)

    # Plot the trajectory
    plt.plot(x_trajectory, y_trajectory, marker='o')  # 'o' indicates points on the trajectory
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Trajectory of the Object')
    plt.grid(True)  # Add a grid
    plt.show()