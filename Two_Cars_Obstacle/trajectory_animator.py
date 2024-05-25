import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from parameters import Params

class TrajectoryAnimator:
    def __init__(self, x_trajectory_1, y_trajectory_1, x_end_1, y_end_1, x_trajectory_2, y_trajectory_2, x_end_2, y_end_2, xobs, yobs):
        #self.obstacle = obstacle
        self.x_trajectory_1 = x_trajectory_1
        self.y_trajectory_1 = y_trajectory_1
        self.x_end_1 = x_end_1
        self.y_end_1 = y_end_1
        self.x_trajectory_2 = x_trajectory_2
        self.y_trajectory_2 = y_trajectory_2
        self.x_end_2 = x_end_2
        self.y_end_2 = y_end_2

        # Car radius based on collision safety parameter
        car_radius = Params['car_size']

        # Create a figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_title('Trajectory of the Objects')
        self.ax.grid(True)

        # Calculate the size of the markers
        # Get the figure DPI (dots per inch)
        dpi = self.fig.dpi
        # Convert radius in data units to radius in points
        radius_in_points = car_radius*1.2 * dpi / self.fig.get_size_inches()[0]
        car_area = np.pi * (radius_in_points) ** 2  # Convert radius to area in points squared

        # Define the car icons with specified area
        self.car_1 = self.ax.scatter([], [], s=car_area, marker='o', color='blue', label='Car 1')
        self.car_2 = self.ax.scatter([], [], s=car_area, marker='o', color='green', label='Car 2')

        # Initialize the collision marker
        self.collision_marker, = self.ax.plot([], [], 'ro', markersize=10, label='Collision', visible=False)

        # Set equal aspect ratio
        self.ax.set_aspect('equal')

        # Plot start and end points
        self.ax.plot(x_trajectory_1[0].detach().clone().numpy(), y_trajectory_1[0].detach().clone().numpy(), 'b', marker='x', label='Start 1')
        self.ax.plot(x_end_1.detach().clone().numpy(), y_end_1.detach().clone().numpy(), 'r', marker='*', label='End 1')
        self.ax.plot(x_trajectory_2[0].detach().clone().numpy(), y_trajectory_2[0].detach().clone().numpy(), 'g', marker='x', label='Start 2')
        self.ax.plot(x_end_2.detach().clone().numpy(), y_end_2.detach().clone().numpy(), 'y', marker='*', label='End 2')

        # Plot the environment limits
        x_min, x_max = Params["Environment_limits"][0]
        y_min, y_max = Params["Environment_limits"][1]
        self.ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], 'k')

        # Initialize the car's positions
        self.car_1.set_offsets([[x_trajectory_1[0].item(), y_trajectory_1[0].item()]])
        self.car_2.set_offsets([[x_trajectory_2[0].item(), y_trajectory_2[0].item()]])

        # Add legend and move it to the right
        #self.ax.legend(loc='upper right')

        # Draw the obstacle
        self.obstacle = plt.Circle((xobs, yobs), Params['obssize'], color='red', fill=False, label='Obstacle')
        self.ax.add_patch(self.obstacle)

    def update(self, frame):
        # Update the cars' positions
        self.car_1.set_offsets([[self.x_trajectory_1[frame].item(), self.y_trajectory_1[frame].item()]])
        self.car_2.set_offsets([[self.x_trajectory_2[frame].item(), self.y_trajectory_2[frame].item()]])

        # Check for collision
        dist = np.sqrt((self.x_trajectory_1[frame].item() - self.x_trajectory_2[frame].item()) ** 2 +
                       (self.y_trajectory_1[frame].item() - self.y_trajectory_2[frame].item()) ** 2)
        if dist < (2*Params['car_size']):
            self.collision_marker.set_data([(self.x_trajectory_1[frame].item() + self.x_trajectory_2[frame].item()) / 2],
                                           [(self.y_trajectory_1[frame].item() + self.y_trajectory_2[frame].item()) / 2])
            self.collision_marker.set_visible(True)
        else:
            self.collision_marker.set_visible(False)

        return self.car_1, self.car_2, self.collision_marker

    def animate(self):
        # Create the animation
        self.ani = FuncAnimation(self.fig, self.update, frames=len(self.x_trajectory_1), blit=True, interval=70)
        # Show the animation
        plt.show()