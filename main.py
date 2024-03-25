import torch
from torch.optim import Adam
from network import create_nn
from dynamics import ObjectMovement
import matplotlib.pyplot as plt
from obstacle_generator import Obstacle_generator
#Setting the defined area of where the trajectory can be made
#The mac and min will be the defined interval of the x- and y-axis

#Initialize obstacles
obstacle_generator = Obstacle_generator()
#Generate obstacle
obstacle = obstacle_generator.generate_obstacle()

max_value = 10
min_value = -10

#Input sample
x_start = torch.rand(1) * min_value
y_start = torch.rand(1) * min_value
x_end = torch.rand(1) * max_value
y_end = torch.rand(1) * max_value
x_end = x_end.clone().detach().requires_grad_(True)
y_end = y_end.clone().detach().requires_grad_(True)
speed_start = 0
angle_start = 0
test1 = x_start

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
    x = x_start.clone()
    y = y_start.clone()
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
        x, y, speed, angle = obj.move_object(delta_speed.item(), delta_angle.item())

        # Update the input sample
        input_sample = torch.tensor([x, y, x_end, y_end, speed, angle])
        input_sample = input_sample.float()

        # Keep the positions for plotting
        x_trajectory=torch.cat((x_trajectory,x),0)
        y_trajectory=torch.cat((y_trajectory,y),0)

        # update loss
        loss += criterion(x, y, x_end, y_end)

    loss /= TrajectoryLength
    # Perform gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Total Loss:", loss)

    # Plot the trajectory
    plt.plot(x_trajectory, y_trajectory, marker='o')  # 'o' indicates points on the trajectory
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Trajectory of the Object')
    plt.grid(True)
    plt.show()