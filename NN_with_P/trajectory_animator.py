import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from parameters import Params

class TrajectoryAnimator:
    def __init__(self, obstacle, x_trajectory, y_trajectory, x_end, y_end):
        self.obstacle = obstacle
        self.x_trajectory = x_trajectory
        self.y_trajectory = y_trajectory
        self.x_end = x_end
        self.y_end = y_end

        # Create a figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_title('Trajectory of the Object')
        self.ax.grid(True)

        # Define the car icon
        self.car = self.ax.scatter([], [], marker='o', color='blue')

        # Define the obstacle
        self.circle = plt.Circle((obstacle[0], obstacle[1]), Params['obssize'], color='red', fill=False)
        self.ax.add_patch(self.circle)

        # Set equal aspect ratio
        self.ax.set_aspect('equal')

        # plot start and end points
        plt.plot(x_trajectory[0].detach().clone().numpy(), y_trajectory[0].detach().clone().numpy(),'b',marker='x')
        plt.plot(x_end.detach().clone().numpy(),y_end.detach().clone().numpy(),'r',marker='*')

        # Plot the environment limits
        x_min, x_max = Params["Environment_limits"][0]
        y_min, y_max = Params["Environment_limits"][1]
        self.ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], 'k')

        # Initialize the car's position
        self.car.set_offsets([[x_trajectory[0].item(), y_trajectory[0].item()]])

    def update(self, frame):
        # Update the car's position
        self.car.set_offsets([[self.x_trajectory[frame].item(), self.y_trajectory[frame].item()]])
        return self.car,

    def animate(self):
        # Create the animation
        self.ani = FuncAnimation(self.fig, self.update, frames=len(self.x_trajectory), blit=True)
        # Show the animation
        plt.show()