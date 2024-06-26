import torch  # Importing the PyTorch library for tensor computations and deep learning
from torch.optim import Adam  
from network import create_nn 
from dynamics import ObjectMovement  
from loss_complete import loss_fn 
from parameters import Params 
import math 
from PD_controller import controller  
from trajectory_animator import TrajectoryAnimator  
from position_validator import positions_okay  

# Number of cars
num_cars = Params['number_of_cars']

# Define the environment boundaries for car starting and ending positions
min_value_x = Params['Environment_limits'][0][0] + Params['start_radius'] + Params['car_size']
max_value_x = Params['Environment_limits'][0][1] - Params['start_radius'] - Params['car_size']
min_value_y = Params['Environment_limits'][1][0] + Params['start_radius'] + Params['car_size']
max_value_y = Params['Environment_limits'][1][1] - Params['start_radius'] - Params['car_size']

# Initialize start and end positions for all cars, ensuring they are valid
while True :
    x_start = [torch.rand(1) * (max_value_x - min_value_x) + min_value_x for _ in range(num_cars)]
    y_start = [torch.rand(1) * (max_value_y - min_value_y) + min_value_y for _ in range(num_cars)]
    x_end = [torch.rand(1) * (max_value_x - min_value_x) + min_value_x for _ in range(num_cars)]
    y_end = [torch.rand(1) * (max_value_y - min_value_y) + min_value_y for _ in range(num_cars)]
    if (positions_okay(x_start, y_start, x_end, y_end)) : break

# Prepare end positions to require gradients for optimization
x_end = [x.clone().detach().requires_grad_(True) for x in x_end]
y_end = [y.clone().detach().requires_grad_(True) for y in y_end]

# Initial speed and random starting angles for all cars
speed_start = 0
angle_start = [torch.rand(1)*2*torch.pi for i in range(num_cars)]

# Initiate controller
controller = controller()

TrajectoryLength = Params['trajectory_length']

# Create neural network models for all cars
models = [create_nn() for _ in range(num_cars)]

# Define optimizers for all car models
optimizers = [Adam(model.parameters(), lr=0.001) for model in models]
criterion = loss_fn()  # Instantiate the loss function

#torch.autograd.set_detect_anomaly(True)

for i in range(10001):
    # Generate random samples around the starting points
    radii = [torch.rand(1) * Params['start_radius'] for _ in range(num_cars)]
    thetas = [torch.rand(1) * 2 * math.pi for _ in range(num_cars)]

    # Initialize positions, speeds, and angles
    positions = [(x_start[j] + radii[j] * torch.cos(thetas[j]), y_start[j] + radii[j] * torch.sin(thetas[j])) for j in range(num_cars)]
    speeds = [speed_start for _ in range(num_cars)]
    angles = [angle_start[j] for j in range(num_cars)]

    # Prepare input samples for the neural networks
    input_samples = [torch.tensor([*positions[j], x_end[j], y_end[j], speeds[j], angles[j]]) for j in range(num_cars)]

    # Initialize trajectories to store positions
    x_trajectories = [torch.tensor([pos[0]], requires_grad=True) for pos in positions]
    y_trajectories = [torch.tensor([pos[1]], requires_grad=True) for pos in positions]

    # Initiate loss
    losses = [0 for _ in range(num_cars)]

    # Simulate trajectory for each car 
    for j in range(TrajectoryLength):
        delta_speeds_nn = []
        delta_angles_nn = []
        delta_speeds_P = []
        delta_angles_P = []

       
        for car_idx in range(num_cars):
             # Get neural network outputs for speed and angle changes
            delta_speed_nn, delta_angle_nn = models[car_idx](input_samples[car_idx])
            # Get PD controller outputs for speed and angle changes
            delta_speed_P, delta_angle_P = controller.forward(positions[car_idx][0], positions[car_idx][1], x_end[car_idx], y_end[car_idx], speeds[car_idx], angles[car_idx])
            delta_speeds_nn.append(delta_speed_nn)
            delta_angles_nn.append(delta_angle_nn)
            delta_speeds_P.append(delta_speed_P)
            delta_angles_P.append(delta_angle_P)

        for car_idx in range(num_cars):
            # Update object's position based on neural network and proportional controller output
            obj = ObjectMovement(positions[car_idx][0], positions[car_idx][1], speeds[car_idx], angles[car_idx])
            new_pos_x, new_pos_y, new_speed, new_angle = obj.move_object(delta_speeds_nn[car_idx], delta_angles_nn[car_idx], delta_speeds_P[car_idx], delta_angles_P[car_idx])
            positions[car_idx] = (new_pos_x, new_pos_y)
            speeds[car_idx] = new_speed
            angles[car_idx] = new_angle

             # Update the input sample with the new position, speed, and angle
            input_samples[car_idx] = torch.tensor([new_pos_x, new_pos_y, x_end[car_idx], y_end[car_idx], new_speed, new_angle]).float()

            # Keep the positions for plotting the trajectory
            x_trajectories[car_idx] = torch.cat((x_trajectories[car_idx], new_pos_x), 0)
            y_trajectories[car_idx] = torch.cat((y_trajectories[car_idx], new_pos_y), 0)

        # Update loss for each car considering all other cars
        for car_idx in range(num_cars):
            for other_car_idx in range(num_cars):
                losses[car_idx] += criterion(positions[car_idx][0], positions[car_idx][1], positions[other_car_idx][0].item(), positions[other_car_idx][1].item(), x_end[car_idx], y_end[car_idx], car_idx, other_car_idx)

    # Normalize losses by the trajectory length
    losses = [loss / TrajectoryLength for loss in losses]

     # Perform gradient descent to update the neural network weights
    for car_idx in range(num_cars):
        optimizers[car_idx].zero_grad()  # Clear the gradients
        losses[car_idx].backward(retain_graph=True) # Backpropagate the loss
        optimizers[car_idx].step()  # Update the weights

    # Print the iteration number and the total loss for each car
    print("Iteration: {}".format(i))
    for car_idx in range(num_cars):
        print("Car {}: Total Loss: {:.4f}".format(car_idx + 1, losses[car_idx].item()))

    # Every 100 iterations, animate the trajectory
    if i % 100 == 0:
        animator = TrajectoryAnimator(x_trajectories, y_trajectories, x_end, y_end)
        animator.animate()






   
