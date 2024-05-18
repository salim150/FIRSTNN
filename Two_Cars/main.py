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
x_start_1 = torch.rand(1) * (Params['Environment_limits'][0][0] + Params['start_radius'] + Params['car_size'])
y_start_1 = torch.rand(1) * (Params['Environment_limits'][1][0] + Params['start_radius'] + Params['car_size'])
x_end_1 = torch.rand(1) * (Params['Environment_limits'][0][1] - Params['car_size'])
y_end_1 = torch.rand(1) * (Params['Environment_limits'][1][1] - Params['car_size'])
x_end_1 = x_end_1.clone().detach().requires_grad_(True)
y_end_1 = y_end_1.clone().detach().requires_grad_(True)

x_start_2 = torch.rand(1) * (Params['Environment_limits'][0][1] - Params['start_radius'] - Params['car_size'])
y_start_2 = torch.rand(1) * (Params['Environment_limits'][1][0] + Params['start_radius'] + Params['car_size'])
x_end_2 = torch.rand(1) * (Params['Environment_limits'][0][0] + Params['car_size'])
y_end_2 = torch.rand(1) * (Params['Environment_limits'][1][1] - Params['car_size'])
x_end_2 = x_end_2.clone().detach().requires_grad_(True)
y_end_2 = y_end_2.clone().detach().requires_grad_(True)

speed_start = 0
angle_start_1 = 0
angle_start_2 = torch.pi

# Initialize obstacles
obstacle_generator = Obstacle_generator()
# Generate obstacle
obstacle = obstacle_generator.generate_obstacle(x_start_1,y_start_1,x_end_1,y_end_1,x_start_2,y_start_2,x_end_2,y_end_2)

# Initiate proportionnal controller
prop_controller = Prop_controller()

TrajectoryLength = 20

# Create neural network model
model_1 = create_nn()
model_2 = create_nn()

# Define optimizer
optimizer1 = Adam(model_1.parameters(), lr=0.001)
optimizer2 = Adam(model_2.parameters(), lr=0.001)
criterion = loss_fn()

#torch.autograd.set_detect_anomaly(True)

for i in range(1001):
    # Generate a random sample around the starting point
    radius_1 = torch.rand(1) * Params['start_radius']
    theta_1 = torch.rand(1) * 2 * math.pi
    radius_2 = torch.rand(1) * Params['start_radius']
    theta_2 = torch.rand(1) * 2 * math.pi

    x_1 = x_start_1 + radius_1 * torch.cos(theta_1)
    y_1 = y_start_1 + radius_1 * torch.sin(theta_1)
    speed_1 = speed_start
    angle_1 = angle_start_1
    x_2 = x_start_2 + radius_2 * torch.cos(theta_2)
    y_2 = y_start_2 + radius_2 * torch.sin(theta_2)
    speed_2 = speed_start
    angle_2 = angle_start_2
    
    input_sample_1 = torch.tensor([x_1, y_1, x_end_1, y_end_1, speed_1, angle_1])
    input_sample_2 = torch.tensor([x_2, y_2, x_end_2, y_end_2, speed_2, angle_2])


    x_trajectory_1=torch.tensor([x_1], requires_grad = True)
    y_trajectory_1=torch.tensor([y_1], requires_grad = True)
    x_trajectory_2=torch.tensor([x_2], requires_grad = True)
    y_trajectory_2=torch.tensor([y_2], requires_grad = True)

    # initiate loss
    loss_1 = 0
    loss_2 = 0
    
    # Perform trajectory
    for j in range(TrajectoryLength):
        # Call neural network to get desired speed and angle changes
        delta_speed_nn_1, delta_angle_nn_1 = model_1(input_sample_1)
        delta_speed_P_1, delta_angle_P_1 = prop_controller.forward(x_1, y_1, x_end_1, y_end_1, speed_1, angle_1)
        delta_speed_nn_2, delta_angle_nn_2 = model_2(input_sample_2)
        delta_speed_P_2, delta_angle_P_2 = prop_controller.forward(x_2, y_2, x_end_2, y_end_2, speed_2, angle_2)

        # Update object's position
        obj_1 = ObjectMovement(x_1, y_1, speed_1, angle_1)
        x_1, y_1, speed_1, angle_1 = obj_1.move_object(delta_speed_nn_1, delta_angle_nn_1, delta_speed_P_1, delta_angle_P_1)

        obj_2 = ObjectMovement(x_2, y_2, speed_2, angle_2)
        x_2, y_2, speed_2, angle_2 = obj_2.move_object(delta_speed_nn_2, delta_angle_nn_2, delta_speed_P_2, delta_angle_P_2)

        # Update the input sample
        input_sample_1 = torch.tensor([x_1, y_1, x_end_1, y_end_1, speed_1, angle_1])
        input_sample_1 = input_sample_1.float()
        input_sample_2 = torch.tensor([x_2, y_2, x_end_2, y_end_2, speed_2, angle_2])
        input_sample_2 = input_sample_2.float()

        # Keep the positions for plotting
        x_trajectory_1=torch.cat((x_trajectory_1,x_1),0)
        y_trajectory_1=torch.cat((y_trajectory_1,y_1),0)
        x_trajectory_2=torch.cat((x_trajectory_2,x_2),0)
        y_trajectory_2=torch.cat((y_trajectory_2,y_2),0)

        # update loss
        loss_1 += criterion(x_1, y_1, x_2, y_2, obstacle[0], obstacle[1], x_end_1, y_end_1, j)
        loss_2 += criterion(x_2, y_2, x_1, y_1, obstacle[0], obstacle[1], x_end_2, y_end_2, j)


    loss_1 /= TrajectoryLength
    loss_2 /= TrajectoryLength

    # Perform gradient descent
    optimizer1.zero_grad()
    loss_1.backward(retain_graph=True)
    optimizer1.step()


    optimizer2.zero_grad()
    loss_2.backward()
    optimizer2.step()

    
    print("Total Loss 1: {:.4f}, Total Loss 2: {:.4f}, Iteration: {}".format(loss_1.item(), loss_2.item(), i))
    if (i%50 == 0) :
        animator = TrajectoryAnimator(x_trajectory_1, y_trajectory_1, x_end_1, y_end_1, x_trajectory_2, y_trajectory_2, x_end_2, y_end_2, obstacle[0], obstacle[1])
        animator.animate()