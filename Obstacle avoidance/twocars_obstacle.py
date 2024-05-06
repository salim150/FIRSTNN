import torch
from torch.optim import Adam
from network import create_nn
from dynamics import ObjectMovement
import matplotlib.pyplot as plt
from obstacle_generator2 import Obstacle_generator
from loss_complete import loss_fn
from parameters import Params

# Setting the defined area of where the trajectory can be made
# The max and min will be the defined interval of the x- and y-axis

# Input sample (making one for 2 different cars)
x_start_car1 = torch.rand(1) * Params['Environment_limits'][0][0]
y_start_car1 = torch.rand(1) * Params['Environment_limits'][1][0]
x_end_car1 = torch.rand(1) * Params['Environment_limits'][0][1]
y_end_car1 = torch.rand(1) * Params['Environment_limits'][1][1]
x_end_car1 = x_end_car1.clone().detach().requires_grad_(True)
y_end_car1 = y_end_car1.clone().detach().requires_grad_(True)
speed_start_car1 = 0
angle_start_car1 = 0

# Same for car 2
x_start_car2 = torch.rand(1) * Params['Environment_limits'][0][0]
y_start_car2 = torch.rand(1) * Params['Environment_limits'][1][0]
x_end_car2 = torch.rand(1) * Params['Environment_limits'][0][1]
y_end_car2 = torch.rand(1) * Params['Environment_limits'][1][1]
x_end_car2 = x_end_car2.clone().detach().requires_grad_(True)
y_end_car2 = y_end_car2.clone().detach().requires_grad_(True)
speed_start_car2 = 0
angle_start_car2 = 0

# Initialize obstacles
obstacle_generator = Obstacle_generator()
# Generate obstacle
obstacle = obstacle_generator.generate_obstacle(x_start_car1, y_start_car1, x_end_car1, y_end_car1)

TrajectoryLength = 20

# Create neural network model
model = create_nn()

# Define optimizer
optimizer = Adam(model.parameters(), lr=0.1)
criterion1 = loss_fn()
criterion2 = loss_fn()

torch.autograd.set_detect_anomaly(True)

# Input samples for both cars
input_sample_car1 = torch.tensor([x_start_car1, y_start_car1, x_end_car1, y_end_car1, speed_start_car1, angle_start_car1])
input_sample_car2 = torch.tensor([x_start_car2, y_start_car2, x_end_car2, y_end_car2, speed_start_car2, angle_start_car2])

print(f"The input tensor for car 1 is: ")
print(input_sample_car1)
print(f"The input tensor for car 2 is: ")
print(input_sample_car2)

for i in range(1001):

    # Starting with initial position and speed for car 1
    x_car1 = x_start_car1.clone()
    y_car1 = y_start_car1.clone()
    speed_car1 = speed_start_car1
    angle_car1 = angle_start_car1

    # Starting with initial position and speed for car 2
    x_car2 = x_start_car2.clone()
    y_car2 = y_start_car2.clone()
    speed_car2 = speed_start_car2
    angle_car2 = angle_start_car2

    # Input samples for both cars
    input_sample_car1 = torch.tensor([x_car1, y_car1, x_end_car1, y_end_car1, speed_car1, angle_car1])
    input_sample_car2 = torch.tensor([x_car2, y_car2, x_end_car2, y_end_car2, speed_car2, angle_car2])

    x_trajectory_car1 = torch.tensor([x_car1], requires_grad=True)
    y_trajectory_car1 = torch.tensor([y_car1], requires_grad=True)

    x_trajectory_car2 = torch.tensor([x_car2], requires_grad=True)
    y_trajectory_car2 = torch.tensor([y_car2], requires_grad=True)

    # initiate loss for both cars
    loss_car1 = 0
    loss_car2 = 0

    # Perform trajectory for both cars
    for j in range(TrajectoryLength):
        # Call neural network to get desired speed and angle changes for car 1
        delta_speed_car1, delta_angle_car1 = model(input_sample_car1)

        # Call neural network to get desired speed and angle changes for car 2
        delta_speed_car2, delta_angle_car2 = model(input_sample_car2)

        # Update object's position for car 1
        obj_car1 = ObjectMovement(x_car1, y_car1, speed_car1, angle_car1)
        x_car1, y_car1, speed_car1, angle_car1 = obj_car1.move_object(delta_speed_car1, delta_angle_car1)

        # Update object's position for car 2
        obj_car2 = ObjectMovement(x_car2, y_car2, speed_car2, angle_car2)
        x_car2, y_car2, speed_car2, angle_car2 = obj_car2.move_object(delta_speed_car2, delta_angle_car2)

        # Update the input samples for both cars
        input_sample_car1 = torch.tensor([x_car1, y_car1, x_end_car1, y_end_car1, speed_car1, angle_car1])
        input_sample_car1 = input_sample_car1.float()

        input_sample_car2 = torch.tensor([x_car2, y_car2, x_end_car2, y_end_car2, speed_car2, angle_car2])
        input_sample_car2 = input_sample_car2.float()

        # Keep the positions for plotting for both cars
        x_trajectory_car1 = torch.cat((x_trajectory_car1, x_car1), 0)
        y_trajectory_car1 = torch.cat((y_trajectory_car1, y_car1), 0)

        x_trajectory_car2 = torch.cat((x_trajectory_car2, x_car2), 0)
        y_trajectory_car2 = torch.cat((y_trajectory_car2, y_car2), 0)

        # update loss for both cars
        loss_car1 += criterion1(x_car1, y_car1, obstacle[0], obstacle[1], x_end_car1, y_end_car1, j)
        loss_car2 += criterion2(x_car2, y_car2, obstacle[0], obstacle[1], x_end_car2, y_end_car2, j)

    loss_car1 /= TrajectoryLength
    loss_car2 /= TrajectoryLength

    # Perform gradient descent for both cars
    optimizer.zero_grad()
    loss_car1.backward()
    loss_car2.backward()
    optimizer.step()

    # Print losses for both cars
    print("Total Loss for car 1:", loss_car1)  # Corrected: added .item()
    print("Total Loss for car 2:", loss_car2)  # Corrected: added .item()
    if (i%200 == 0) :
        # Plot the trajectory
        plt.plot(x_trajectory_car1.detach().clone().numpy(), y_trajectory_car1.detach().clone().numpy(), marker='o', label = 'Car 1')  # 'o' indicates points on the trajectory 
        plt.plot(x_trajectory_car2.detach().clone().numpy(), y_trajectory_car2.detach().clone().numpy(), marker='o', label = 'Car 2')
        plt.plot(x_start_car1.detach().clone().numpy(),y_start_car1.detach().clone().numpy(),'r',marker='x', label='Car 1 Start')
        plt.plot(x_start_car2.detach().clone().numpy(),y_start_car2.detach().clone().numpy(),'b',marker='x', label='Car 2 Start')
        plt.plot(x_end_car1.detach().clone().numpy(),y_end_car1.detach().clone().numpy(),'r',marker='*', label='Car 1 Goal')
        plt.plot(x_end_car2.detach().clone().numpy(),y_end_car2.detach().clone().numpy(),'b',marker='*', label='Car 2 Goal')
        # Generate and plot the obstacle (the initialization of the obstacle was done before the loop)
        #plt.gca().add_patch(obstacle)
        x_min, x_max = Params["Environment_limits"][0]
        y_min, y_max = Params["Environment_limits"][1]
         # Generate and plot the obstacle
        plt.plot(obstacle[0].item(), obstacle[1].item(), 'ko', label='Obstacle')
        plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], 'k')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Trajectory of the Object')
        plt.grid(True)
        plt.show()