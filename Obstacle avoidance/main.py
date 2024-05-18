import torch
from torch.optim import Adam
from network_1 import create_nn_1
from network_2 import create_nn_2
from network_3 import create_nn_3
from network_4 import create_nn_4
from dynamics import ObjectMovement
import matplotlib.pyplot as plt
from obstacle_generator2 import Obstacle_generator
from loss_complete import loss_fn
from parameters import Params
import math
from P_controller import Prop_controller
from trajectory_animator import TrajectoryAnimator

#Setting the defined area of where the trajectory can be made
#The mac and min will be the defined interval of the x- and y-axis

#Input sample
x_start_1 = torch.rand(1) * (Params['Environment_limits'][0][0] + Params['start_radius'])
y_start_1 = torch.rand(1) * (Params['Environment_limits'][1][0] + Params['start_radius'])
x_end_1 = torch.rand(1) * Params['Environment_limits'][0][1]
y_end_1 = torch.rand(1) * Params['Environment_limits'][1][1]
x_end_1 = x_end_1.clone().detach().requires_grad_(True)
y_end_1 = y_end_1.clone().detach().requires_grad_(True)

x_start_2 = torch.rand(1) * (Params['Environment_limits'][0][1] + Params['start_radius'])
y_start_2 = torch.rand(1) * (Params['Environment_limits'][1][0] + Params['start_radius'])
x_end_2 = torch.rand(1) * Params['Environment_limits'][0][0]
y_end_2 = torch.rand(1) * Params['Environment_limits'][1][1]
x_end_2 = x_end_2.clone().detach().requires_grad_(True)
y_end_2 = y_end_2.clone().detach().requires_grad_(True)

x_start_3 = torch.rand(1) * (Params['Environment_limits'][0][1] + Params['start_radius'])
y_start_3 = torch.rand(1) * (Params['Environment_limits'][1][0] + Params['start_radius'])
x_end_3 = torch.rand(1) * Params['Environment_limits'][0][0]
y_end_3 = torch.rand(1) * Params['Environment_limits'][1][1]
x_end_3 = x_end_3.clone().detach().requires_grad_(True)
y_end_3 = y_end_3.clone().detach().requires_grad_(True)

x_start_4 = torch.rand(1) * (Params['Environment_limits'][0][1] + Params['start_radius'])
y_start_4 = torch.rand(1) * (Params['Environment_limits'][1][0] + Params['start_radius'])
x_end_4 = torch.rand(1) * Params['Environment_limits'][0][0]
y_end_4 = torch.rand(1) * Params['Environment_limits'][1][1]
x_end_4 = x_end_4.clone().detach().requires_grad_(True)
y_end_4 = y_end_4.clone().detach().requires_grad_(True)

speed_start = 0
angle_start = 0

# Initialize obstacles
obstacle_generator = Obstacle_generator()
# Generate obstacle
#obstacle = obstacle_generator.generate_obstacle(x_start, y_start, x_end, y_end)

# Initiate proportionnal controller
prop_controller = Prop_controller()

TrajectoryLength = 20

# Create neural network model
model_1 = create_nn_1()
model_2 = create_nn_2()
model_3 = create_nn_3()
model_4 = create_nn_4()

# Define optimizer
optimizer = Adam(list(model_1.parameters()) + list(model_2.parameters()) + list(model_3.parameters()) + list(model_4.parameters()), lr=0.001)
criterion = loss_fn()

torch.autograd.set_detect_anomaly(True)

input_sample_1 = torch.tensor([x_start_1, y_start_1, x_end_1, y_end_1, speed_start, angle_start])
input_sample_2 = torch.tensor([x_start_2, y_start_2, x_end_2, y_end_2, speed_start, angle_start])
input_sample_3 = torch.tensor([x_start_3, y_start_3, x_end_3, y_end_3, speed_start, angle_start])
input_sample_4 = torch.tensor([x_start_4, y_start_4, x_end_4, y_end_4, speed_start, angle_start])

for i in range(1001):
    # Generate a random sample around the starting point
    radius_1 = torch.rand(1) * Params['start_radius']
    theta_1 = torch.rand(1) * 2 * math.pi
    radius_2 = torch.rand(1) * Params['start_radius']
    theta_2 = torch.rand(1) * 2 * math.pi
    radius_3 = torch.rand(1) * Params['start_radius']
    theta_3 = torch.rand(1) * 2 * math.pi
    radius_4 = torch.rand(1) * Params['start_radius']
    theta_4 = torch.rand(1) * 2 * math.pi

    x_1 = x_start_1 + radius_1 * torch.cos(theta_1)
    y_1 = y_start_1 + radius_1 * torch.sin(theta_1)
    speed_1 = speed_start
    angle_1 = angle_start

    x_2 = x_start_2 + radius_2 * torch.cos(theta_2)
    y_2 = y_start_2 + radius_2 * torch.sin(theta_2)
    speed_2 = speed_start
    angle_2 = angle_start

    x_3 = x_start_3 + radius_3 * torch.cos(theta_3)
    y_3 = y_start_3 + radius_3 * torch.sin(theta_3)
    speed_3 = speed_start
    angle_3 = angle_start

    x_4 = x_start_4 + radius_4 * torch.cos(theta_4)
    y_4 = y_start_4 + radius_4 * torch.sin(theta_4)
    speed_4 = speed_start
    angle_4 = angle_start
    
    input_sample_1 = torch.tensor([x_1, y_1, x_end_1, y_end_1, speed_1, angle_1])
    input_sample_2 = torch.tensor([x_2, y_2, x_end_2, y_end_2, speed_2, angle_2])
    input_sample_3 = torch.tensor([x_3, y_3, x_end_3, y_end_3, speed_3, angle_3])
    input_sample_4 = torch.tensor([x_4, y_4, x_end_4, y_end_4, speed_4, angle_4])

    x_trajectory_1 = torch.tensor([x_1], requires_grad=True)
    y_trajectory_1 = torch.tensor([y_1], requires_grad=True)
    x_trajectory_2 = torch.tensor([x_2], requires_grad=True)
    y_trajectory_2 = torch.tensor([y_2], requires_grad=True)
    x_trajectory_3 = torch.tensor([x_3], requires_grad=True)
    y_trajectory_3 = torch.tensor([y_3], requires_grad=True)
    x_trajectory_4 = torch.tensor([x_4], requires_grad=True)
    y_trajectory_4 = torch.tensor([y_4], requires_grad=True)

    # Initiate loss
    loss_1 = 0
    loss_2 = 0
    loss_3 = 0
    loss_4 = 0

    # Perform trajectory
    for j in range(TrajectoryLength):
        # Call neural network to get desired speed and angle changes
        delta_speed_nn_1, delta_angle_nn_1 = model_1(input_sample_1)
        delta_speed_P_1, delta_angle_P_1 = prop_controller.forward(x_1, y_1, x_end_1, y_end_1, speed_1, angle_1)
        delta_speed_nn_2, delta_angle_nn_2 = model_2(input_sample_2)
        delta_speed_P_2, delta_angle_P_2 = prop_controller.forward(x_2, y_2, x_end_2, y_end_2, speed_2, angle_2)
        delta_speed_nn_3, delta_angle_nn_3 = model_3(input_sample_3)
        delta_speed_P_3, delta_angle_P_3 = prop_controller.forward(x_3, y_3, x_end_3, y_end_3, speed_3, angle_3)
        delta_speed_nn_4, delta_angle_nn_4 = model_4(input_sample_4)
        delta_speed_P_4, delta_angle_P_4 = prop_controller.forward(x_4, y_4, x_end_4, y_end_4, speed_4, angle_4)

        # Update object's position
        obj_1 = ObjectMovement(x_1, y_1, speed_1, angle_1)
        x_1, y_1, speed_1, angle_1 = obj_1.move_object(delta_speed_nn_1, delta_angle_nn_1, delta_speed_P_1, delta_angle_P_1)

        obj_2 = ObjectMovement(x_2, y_2, speed_2, angle_2)
        x_2, y_2, speed_2, angle_2 = obj_2.move_object(delta_speed_nn_2, delta_angle_nn_2, delta_speed_P_2, delta_angle_P_2)

        obj_3 = ObjectMovement(x_3, y_3, speed_3, angle_3)
        x_3, y_3, speed_3, angle_3 = obj_3.move_object(delta_speed_nn_3, delta_angle_nn_3, delta_speed_P_3, delta_angle_P_3)

        obj_4 = ObjectMovement(x_4, y_4, speed_4, angle_4)
        x_4, y_4, speed_4, angle_4 = obj_4.move_object(delta_speed_nn_4, delta_angle_nn_4, delta_speed_P_4, delta_angle_P_4)

        # Update the input sample
        input_sample_1 = torch.tensor([x_1, y_1, x_end_1, y_end_1, speed_1, angle_1]).float()
        input_sample_2 = torch.tensor([x_2, y_2, x_end_2, y_end_2, speed_2, angle_2]).float()
        input_sample_3 = torch.tensor([x_3, y_3, x_end_3, y_end_3, speed_3, angle_3]).float()
        input_sample_4 = torch.tensor([x_4, y_4, x_end_4, y_end_4, speed_4, angle_4]).float()

        # Keep the positions for plotting
        x_trajectory_1 = torch.cat((x_trajectory_1, x_1), 0)
        y_trajectory_1 = torch.cat((y_trajectory_1, y_1), 0)
        x_trajectory_2 = torch.cat((x_trajectory_2, x_2), 0)
        y_trajectory_2 = torch.cat((y_trajectory_2, y_2), 0)
        x_trajectory_3 = torch.cat((x_trajectory_3, x_3), 0)
        y_trajectory_3 = torch.cat((y_trajectory_3, y_3), 0)
        x_trajectory_4 = torch.cat((x_trajectory_4, x_4), 0)
        y_trajectory_4 = torch.cat((y_trajectory_4, y_4), 0)

        car_pos_1 = torch.tensor([x_1, y_1])
        goal_pos_1 = torch.tensor([x_end_1, y_end_1])
        other_positions_1 = torch.tensor([(x_2, y_2), (x_3, y_3), (x_4, y_4)])

        car_pos_2 = torch.tensor([x_2, y_2])
        goal_pos_2 = torch.tensor([x_end_2, y_end_2])
        other_positions_2 = torch.tensor([(x_1, y_1), (x_3, y_3), (x_4, y_4)])

        car_pos_3 = torch.tensor([x_3, y_3])
        goal_pos_3 = torch.tensor([x_end_3, y_end_3])
        other_positions_3 = torch.tensor([(x_1, y_1), (x_2, y_2), (x_4, y_4)])

        car_pos_4 = torch.tensor([x_4, y_4])
        goal_pos_4 = torch.tensor([x_end_4, y_end_4])
        other_positions_4 = torch.tensor([(x_1, y_1), (x_2, y_2), (x_3, y_3)])

        loss_1 += criterion(car_pos_1, goal_pos_1, other_positions_1)
        loss_2 += criterion(car_pos_2, goal_pos_2, other_positions_2)
        loss_3 += criterion(car_pos_3, goal_pos_3, other_positions_3)
        loss_4 += criterion(car_pos_4, goal_pos_4, other_positions_4)

    loss_1 /= TrajectoryLength
    loss_2 /= TrajectoryLength
    loss_3 /= TrajectoryLength
    loss_4 /= TrajectoryLength

    # Perform gradient descent
    optimizer.zero_grad()
    loss_1.backward(retain_graph=True)
    loss_2.backward(retain_graph=True)
    loss_3.backward(retain_graph=True)
    loss_4.backward()
    optimizer.step()
    
    print("Total Loss 1:", loss_1.item(),"Total Loss 2:", loss_2.item(), "Total Loss 3:", loss_3.item(), "Total Loss 4:", loss_4.item(), "Iteration:", i)

    if i % 50 == 0:
        animator = TrajectoryAnimator(x_trajectory_1, y_trajectory_1, x_end_1, y_end_1, x_trajectory_2, y_trajectory_2, x_end_2, y_end_2, x_trajectory_3, y_trajectory_3, x_end_3, y_end_3, x_trajectory_4, y_trajectory_4, x_end_4, y_end_4)
        animator.animate()
