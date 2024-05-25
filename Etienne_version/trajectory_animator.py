import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from parameters import Params

class TrajectoryAnimator:
    def __init__(self, obstacle, x_trajectory, y_trajectory, x_end, y_end):
        self.obstacle = obstacle
        self.x_trajectory = x_trajectory
        self.y_trajectory = y_trajectory
        self.x_end = x_end
        self.y_end = y_end

        # Car radius based on collision safety parameter
        car_radius = Params['car_size']

        # Create a figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_title('Trajectory of the Object')
        self.ax.grid(True)

        # Calculate the size of the marker
        dpi = self.fig.dpi
        radius_in_points = car_radius * 1.2 * dpi / self.fig.get_size_inches()[0]
        car_area = np.pi * (radius_in_points) ** 2

        # Define the car icon with specified area
        self.car = self.ax.scatter([], [], s=car_area, marker='o', color='blue')

        # Define the obstacle
        self.circle = plt.Circle((obstacle[0], obstacle[1]), Params['obssize'], color='red', fill=False)
        self.ax.add_patch(self.circle)

        # Set equal aspect ratio
        self.ax.set_aspect('equal')

        # Plot start and end points
        self.ax.plot(x_trajectory[0].detach().clone().numpy(), y_trajectory[0].detach().clone().numpy(), 'b', marker='x', label='Start')
        self.ax.plot(x_end.detach().clone().numpy(), y_end.detach().clone().numpy(), 'r', marker='*', label='End')

        # Plot the environment limits
        x_min, x_max = Params["Environment_limits"][0]
        y_min, y_max = Params["Environment_limits"][1]
        self.ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], 'k')

        # Initialize the car's position
        self.car.set_offsets([[x_trajectory[0].item(), y_trajectory[0].item()]])

        # Add legend and move it to the right
        self.ax.legend(loc='upper right')

    def update(self, frame):
        # Update the car's position
        self.car.set_offsets([[self.x_trajectory[frame].item(), self.y_trajectory[frame].item()]])
        return self.car,

    def animate(self, interval=70):  # Default interval is 70 ms
        # Create the animation
        num_frames = len(self.x_trajectory)
        self.ani = FuncAnimation(self.fig, self.update, frames=num_frames, blit=True, interval=interval)
        # Show the animation
        plt.show()

    def save(self, filename, fps=15, dpi=100):
        # Save the animation as an MP4 file
        writer = FFMpegWriter(fps=fps)
        self.ani.save(filename, writer=writer, dpi=dpi)

# Example usage:
# animator = TrajectoryAnimator(obstacle, x_trajectory, y_trajectory, x_end, y_end)
# animator.animate(interval=70)  # To display the animation
# animator.save('trajectory_animation.mp4', fps=15, dpi=100)  # To save the animation
