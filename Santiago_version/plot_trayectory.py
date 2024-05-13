import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from parameters import Params




def traj_plot(trajectory,obstacle,target,i):
    fig=plt.figure(i+1)
    # Plot arena square
    x_min, x_max = Params["Environment_limits"][0]
    y_min, y_max = Params["Environment_limits"][1]
    plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], 'k')
    plt.plot(trajectory[0,:].detach().numpy(),trajectory[1,:].detach().numpy(), marker='o')
    plt.plot(target[0].clone().detach(),target[1].clone().detach(),'r' ,marker='*')
    plt.plot(trajectory[0,0].clone().detach(),trajectory[1,0].clone().detach(),'b' ,marker='x')

    circle = plt.Circle((obstacle[0].clone().detach().numpy(), obstacle[1].clone().detach().numpy()), Params['obssize'], color='r', fill=False)
    plt.gca().add_patch(circle)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Trajectory of the Object')
    plt.grid(True)
    plt.show()
    return



class TrajectoryAnimator:
    def __init__(self, obstacle, trajectory, xf):
        self.obstacle = obstacle.detach().clone().numpy()
        self.x_trajectory = trajectory[0].detach().clone().numpy()
        self.y_trajectory = trajectory[1].detach().clone().numpy()
        self.x_end = xf[0].detach().clone().numpy()
        self.y_end = xf[1].detach().clone().numpy()

        # Create a figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_title('Trajectory of the Object')
        self.ax.grid(True)

        # Define the car icon
        self.car = self.ax.scatter([], [], marker='>', color='blue')

        # Define the obstacle
        self.circle = plt.Circle((obstacle[0].clone().detach(), obstacle[1].clone().detach()), Params['obssize'], color='red', fill=False)
        self.ax.add_patch(self.circle)

        # Set equal aspect ratio
        self.ax.set_aspect('equal')

        # plot start and end points
        plt.plot(trajectory[0,0].detach().clone().numpy(), trajectory[1,0].detach().clone().numpy(),'b',marker='x')
        plt.plot(xf[0].detach().clone().numpy(),xf[1].detach().clone().numpy(),'r',marker='*')

        # Plot the environment limits
        x_min, x_max = Params["Environment_limits"][0]
        y_min, y_max = Params["Environment_limits"][1]
        self.ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], 'k')

        # Initialize the car's position
        self.car.set_offsets([[trajectory[0,0].clone().detach().item(), trajectory[0,1].clone().detach().item()]])

    def update(self, frame):
        # Update the car's position
        self.car.set_offsets([[self.x_trajectory[frame].item(), self.y_trajectory[frame].item()]])
        return self.car,

    def animate(self):
        # Create the animation
        self.ani = FuncAnimation(self.fig, self.update, frames=len(self.x_trajectory), blit=True)
        # Show the animation
        plt.show()