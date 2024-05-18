import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from parameters import Params

class TrajectoryAnimator:
    def __init__(self, x_trajectory_1, y_trajectory_1, x_end_1, y_end_1, x_trajectory_2, y_trajectory_2, x_end_2, y_end_2):
        #self.obstacle = obstacle
        self.x_trajectory_1 = x_trajectory_1
        self.y_trajectory_1 = y_trajectory_1
        self.x_end_1 = x_end_1
        self.y_end_1 = y_end_1
        self.x_trajectory_2 = x_trajectory_2
        self.y_trajectory_2 = y_trajectory_2
        self.x_end_2 = x_end_2
        self.y_end_2 = y_end_2

        # Create a figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_title('Trajectory of the Objects')
        self.ax.grid(True)

        # Define the car icon
        self.car_1 = self.ax.scatter([], [], marker='o', color='blue')
        self.car_2 = self.ax.scatter([], [], marker='o', color='blue')


        '''# Define the obstacle
        self.circle = plt.Circle((obstacle[0], obstacle[1]), Params['obssize'], color='red', fill=False)
        self.ax.add_patch(self.circle)'''

        # Set equal aspect ratio
        self.ax.set_aspect('equal')

        # plot start and end points
        plt.plot(x_trajectory_1[0].detach().clone().numpy(), y_trajectory_1[0].detach().clone().numpy(),'b',marker='x')
        plt.plot(x_end_1.detach().clone().numpy(),y_end_1.detach().clone().numpy(),'r',marker='*')
        plt.plot(x_trajectory_2[0].detach().clone().numpy(), y_trajectory_2[0].detach().clone().numpy(),'b',marker='x')
        plt.plot(x_end_2.detach().clone().numpy(),y_end_2.detach().clone().numpy(),'r',marker='*')

        # Plot the environment limits
        x_min, x_max = Params["Environment_limits"][0]
        y_min, y_max = Params["Environment_limits"][1]
        self.ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], 'k')

        # Initialize the car's position
        self.car_1.set_offsets([[x_trajectory_1[0].item(), y_trajectory_1[0].item()]])
        self.car_2.set_offsets([[x_trajectory_1[0].item(), y_trajectory_1[0].item()]])

    def update_1(self, frame):
        # Update the car's position
        self.car_1.set_offsets([[self.x_trajectory_1[frame].item(), self.y_trajectory_1[frame].item()]])
        return self.car_1,

    def update_2(self, frame):
        # Update the car's position
        self.car_2.set_offsets([[self.x_trajectory_2[frame].item(), self.y_trajectory_2[frame].item()]])
        return self.car_2,

    def animate(self):
        # Create the animation
        self.ani_1 = FuncAnimation(self.fig, self.update_1, frames=len(self.x_trajectory_1), blit=True)
        self.ani_2 = FuncAnimation(self.fig, self.update_2, frames=len(self.x_trajectory_1), blit=True)
        # Show the animation
        plt.show()