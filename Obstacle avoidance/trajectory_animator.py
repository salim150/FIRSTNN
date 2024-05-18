import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from parameters import Params

class TrajectoryAnimator:
    def __init__(self, x_trajectory_1, y_trajectory_1, x_end_1, y_end_1, x_trajectory_2, y_trajectory_2, x_end_2, y_end_2, x_trajectory_3, y_trajectory_3, x_end_3, y_end_3, x_trajectory_4, y_trajectory_4, x_end_4, y_end_4):
        #self.obstacle = obstacle
        self.x_trajectory_1 = x_trajectory_1
        self.y_trajectory_1 = y_trajectory_1
        self.x_end_1 = x_end_1
        self.y_end_1 = y_end_1
        self.x_trajectory_2 = x_trajectory_2
        self.y_trajectory_2 = y_trajectory_2
        self.x_end_2 = x_end_2
        self.y_end_2 = y_end_2
        self.x_trajectory_3 = x_trajectory_3
        self.y_trajectory_3 = y_trajectory_3
        self.x_end_3 = x_end_3
        self.y_end_3 = y_end_3
        self.x_trajectory_4 = x_trajectory_4
        self.y_trajectory_4 = y_trajectory_4
        self.x_end_4 = x_end_4
        self.y_end_4 = y_end_4

        # Create a figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_title('Trajectory of the Objects')
        self.ax.grid(True)

        # Define the car icons
        self.car_1 = self.ax.scatter([], [], marker='o', color='blue', label='Car 1')
        self.car_2 = self.ax.scatter([], [], marker='o', color='green', label='Car 2')
        self.car_3 = self.ax.scatter([], [], marker='o', color='red', label='Car 3')
        self.car_4 = self.ax.scatter([], [], marker='o', color='green', label='Car 4')
        
        # Set equal aspect ratio
        self.ax.set_aspect('equal')

        # Plot start and end points
        self.ax.plot(x_trajectory_1[0].detach().clone().numpy(), y_trajectory_1[0].detach().clone().numpy(), 'b', marker='x', label='Start 1')
        self.ax.plot(x_end_1.detach().clone().numpy(), y_end_1.detach().clone().numpy(), 'r', marker='*', label='End 1')
        self.ax.plot(x_trajectory_2[0].detach().clone().numpy(), y_trajectory_2[0].detach().clone().numpy(), 'g', marker='x', label='Start 2')
        self.ax.plot(x_end_2.detach().clone().numpy(), y_end_2.detach().clone().numpy(), 'y', marker='*', label='End 2')

        self.ax.plot(x_trajectory_3[0].detach().clone().numpy(), y_trajectory_3[0].detach().clone().numpy(), 'p', marker='x', label='Start 3')
        self.ax.plot(x_end_3.detach().clone().numpy(), y_end_3.detach().clone().numpy(), 'o', marker='*', label='End 3')
        self.ax.plot(x_trajectory_4[0].detach().clone().numpy(), y_trajectory_4[0].detach().clone().numpy(), 'g', marker='x', label='Start 4')
        self.ax.plot(x_end_4.detach().clone().numpy(), y_end_4.detach().clone().numpy(), 'y', marker='*', label='End 4')

        # Plot the environment limits
        x_min, x_max = Params["Environment_limits"][0]
        y_min, y_max = Params["Environment_limits"][1]
        self.ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], 'k')

        # Initialize the car's positions
        self.car_1.set_offsets([[x_trajectory_1[0].item(), y_trajectory_1[0].item()]])
        self.car_2.set_offsets([[x_trajectory_2[0].item(), y_trajectory_2[0].item()]])
        self.car_3.set_offsets([[x_trajectory_3[0].item(), y_trajectory_3[0].item()]])
        self.car_4.set_offsets([[x_trajectory_4[0].item(), y_trajectory_4[0].item()]])
        self.ax.legend(loc='upper right')

    def update(self, frame):
        # Update the cars' positions
        self.car_1.set_offsets([[self.x_trajectory_1[frame].item(), self.y_trajectory_1[frame].item()]])
        self.car_2.set_offsets([[self.x_trajectory_2[frame].item(), self.y_trajectory_2[frame].item()]])
        self.car_3.set_offsets([[self.x_trajectory_3[frame].item(), self.y_trajectory_3[frame].item()]])
        self.car_4.set_offsets([[self.x_trajectory_4[frame].item(), self.y_trajectory_4[frame].item()]])
        return self.car_1, self.car_2, self.car_3, self.car_4

    def animate(self):
        # Create the animation
        self.ani = FuncAnimation(self.fig, self.update, frames=len(self.x_trajectory_1), blit=True)
        # Show the animation
        plt.show()
