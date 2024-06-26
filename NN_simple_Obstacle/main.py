import torch
from torch.optim import Adam
from network import create_nn
from dynamics import ObjectMovement
import matplotlib.pyplot as plt
from obstacle_generator import Obstacle_generator
from loss_complete import loss_fn
from parameters import Params
from trajectory_animator import TrajectoryAnimator
import numpy as np

#Input sample
x_start = torch.rand(1) * (Params['Environment_limits'][0][0] + Params['start_radius'])
y_start = torch.rand(1) * (Params['Environment_limits'][1][0] + Params['start_radius'])
x_end = torch.rand(1) * Params['Environment_limits'][0][1]
y_end = torch.rand(1) * Params['Environment_limits'][1][1]
x_end = x_end.clone().detach().requires_grad_(True)
y_end = y_end.clone().detach().requires_grad_(True)
speed_start = 0
angle_start = 0

#Initialize obstacles
obstacle_generator = Obstacle_generator()
#Generate obstacle
obstacle = obstacle_generator.generate_obstacle(x_start, y_start, x_end, y_end)
obstacle = torch.tensor([0,0])

TrajectoryLength = Params['Trajectory_length']

# Create neural network model
model = create_nn()

# Define optimizer
optimizer = Adam(model.parameters(), lr=0.001)
criterion = loss_fn()

#torch.autograd.set_detect_anomaly(True)

input_sample = torch.tensor([x_start, y_start, x_end, y_end, speed_start, angle_start])

for i in range(10001):
    #generate a random sample around the starting point
    radius = torch.rand(1) * Params['start_radius']
    theta = torch.rand(1) * 2 * torch.pi

    x = x_start + radius * torch.cos(theta)
    y = y_start + radius * torch.sin(theta)
    speed = speed_start
    angle = angle_start
    
    input_sample = torch.tensor([x, y, x_end, y_end, speed, angle])

    x_trajectory=torch.tensor([x], requires_grad = True)
    y_trajectory=torch.tensor([y], requires_grad = True)

    # initiate loss
    loss = 0
    
    # Perform trajectory
    for j in range(TrajectoryLength):
        # Call neural network to get desired speed and angle changes
        delta_speed, delta_angle = model(input_sample)

        # Update object's position
        obj = ObjectMovement(x, y, speed, angle)
        x, y, speed, angle = obj.move_object(delta_speed, delta_angle)

        # Update the input sample
        input_sample = torch.tensor([x, y, x_end, y_end, speed, angle])
        input_sample = input_sample.float()

        # Keep the positions for plotting
        x_trajectory=torch.cat((x_trajectory,x),0)
        y_trajectory=torch.cat((y_trajectory,y),0)

        # update loss
        loss += criterion(x, y,obstacle[0], obstacle[1], x_end, y_end, j)

    loss /= TrajectoryLength
    # Perform gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Total Loss:", loss, "Iteration:", i)
    if (i%500 == 0) :
        animator = TrajectoryAnimator(obstacle, x_trajectory, y_trajectory, x_end, y_end)
        animator.animate()
        # Plot the trajectory
        plt.figure(figsize=(8, 6))
        plt.plot(x_trajectory.detach().numpy(), y_trajectory.detach().numpy(), 'o-', label='Trajectory')
        plt.plot(x_end.detach().numpy(), y_end.detach().numpy(), 'r*', label='End Point')
        circle = plt.Circle((obstacle[0].detach().numpy(), obstacle[1].detach().numpy()), Params['obssize'], color='r', fill=False)
        ax = plt.gca()
        ax.add_patch(circle)
        ax.set_aspect('equal')  # Set aspect ratio to be equal

        # Plot environment limits
        plt.plot([Params['Environment_limits'][0][0], Params['Environment_limits'][0][1]], [Params['Environment_limits'][1][0], Params['Environment_limits'][1][0]], 'k-')  # Bottom
        plt.plot([Params['Environment_limits'][0][0], Params['Environment_limits'][0][1]], [Params['Environment_limits'][1][1], Params['Environment_limits'][1][1]], 'k-')  # Top
        plt.plot([Params['Environment_limits'][0][0], Params['Environment_limits'][0][0]], [Params['Environment_limits'][1][0], Params['Environment_limits'][1][1]], 'k-')  # Left
        plt.plot([Params['Environment_limits'][0][1], Params['Environment_limits'][0][1]], [Params['Environment_limits'][1][0], Params['Environment_limits'][1][1]], 'k-')  # Right

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        #plt.legend()
        plt.grid(True)
        plt.show()