import torch
from torch.optim import Adam
from network import create_nn
from dynamics import ObjectMovement
import matplotlib.pyplot as plt
from obstacle_generator import Obstacle_generator
from loss_complete import loss_fn
from parameters import Params
import math
from P_controller import Prop_controller
from trajectory_animator import TrajectoryAnimator
#Setting the defined area of where the trajectory can be made
#The mac and min will be the defined interval of the x- and y-axis

#Input sample
x_start = torch.rand(1) * (Params['Environment_limits'][0][0] + Params['start_radius'])
y_start = torch.rand(1) * (Params['Environment_limits'][1][0] + Params['start_radius'])
x_end = torch.rand(1) * Params['Environment_limits'][0][1]
y_end = torch.rand(1) * Params['Environment_limits'][1][1]
x_end = x_end.clone().detach().requires_grad_(True)
y_end = y_end.clone().detach().requires_grad_(True)
speed_start = 0
angle_start = 0

# Initialize obstacles
obstacle_generator = Obstacle_generator()
# Generate obstacle
obstacle = obstacle_generator.generate_obstacle(x_start, y_start, x_end, y_end)

# Initiate proportionnal controller
prop_controller = Prop_controller()

TrajectoryLength = 20

# Create neural network model
model = create_nn()

# Define optimizer
optimizer = Adam(model.parameters(), lr=0.001)
criterion = loss_fn()

torch.autograd.set_detect_anomaly(True)

input_sample = torch.tensor([x_start, y_start, x_end, y_end, speed_start, angle_start])

for i in range(1001):
    # Generate a random sample around the starting point
    radius = torch.rand(1) * Params['start_radius']
    theta = torch.rand(1) * 2 * math.pi

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
        delta_speed_nn, delta_angle_nn = model(input_sample)
        delta_speed_P, delta_angle_P = prop_controller.forward(x, y, x_end, y_end, speed, angle)

        # Update object's position
        obj = ObjectMovement(x, y, speed, angle)
        x, y, speed, angle = obj.move_object(delta_speed_nn, delta_angle_nn, delta_speed_P, delta_angle_P)

        # Update the input sample
        input_sample = torch.tensor([x, y, x_end, y_end, speed, angle])
        input_sample = input_sample.float()

        # Keep the positions for plotting
        x_trajectory=torch.cat((x_trajectory,x),0)
        y_trajectory=torch.cat((y_trajectory,y),0)

        # update loss
        loss += criterion(x, y, obstacle[0], obstacle[1], x_end, y_end, j)

    loss /= TrajectoryLength
    # Perform gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Total Loss:", loss, "Iteration:", i)
    if (i%50 == 0) :
        animator = TrajectoryAnimator(obstacle, x_trajectory, y_trajectory, x_end, y_end)
        animator.animate()