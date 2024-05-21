import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from parameters import Params

class TrajectoryAnimator:
    def __init__(self, x_trajectories, y_trajectories, x_ends, y_ends, car_image_path='car_image.png'):
        self.x_trajectories = x_trajectories
        self.y_trajectories = y_trajectories
        self.x_ends = x_ends
        self.y_ends = y_ends
        self.car_image_path = car_image_path

        num_cars = len(x_trajectories)

        # Car radius based on collision safety parameter
        car_radius = Params['car_size']

        # Create a figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_title('Trajectory of the Objects')
        self.ax.grid(True)

        # Load car image
        car_image = plt.imread(self.car_image_path)

        # Plot start and end points
        colors = plt.cm.get_cmap('tab10', num_cars)
        for i in range(num_cars):
            self.ax.plot(x_trajectories[i][0].detach().clone().numpy(), y_trajectories[i][0].detach().clone().numpy(), marker='x', color=colors(i), label=f'Start {i+1}')
            self.ax.plot(x_ends[i].detach().clone().numpy(), y_ends[i].detach().clone().numpy(), marker='*', color=colors(i), label=f'End {i+1}')

        # Plot the environment limits
        x_min, x_max = Params["Environment_limits"][0]
        y_min, y_max = Params["Environment_limits"][1]
        self.ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], 'k')

        # Initialize the car images
        self.car_images = []
        for i in range(num_cars):
            car_icon = OffsetImage(car_image, zoom=car_radius / 50)  # Adjust zoom as needed
            car_annotation = AnnotationBbox(car_icon, (x_trajectories[i][0].item(), y_trajectories[i][0].item()), frameon=False)
            self.ax.add_artist(car_annotation)
            self.car_images.append(car_annotation)

        # Initialize the collision marker
        self.collision_marker, = self.ax.plot([], [], 'ro', markersize=10, label='Collision', visible=False)

        # Set equal aspect ratio
        self.ax.set_aspect('equal')

    def update(self, frame):
        # Update the cars' positions
        for i in range(len(self.car_images)):
            self.car_images[i].xy = (self.x_trajectories[i][frame].item(), self.y_trajectories[i][frame].item())

        # Check for collisions
        collision_detected = False
        for i in range(len(self.car_images)):
            for j in range(i + 1, len(self.car_images)):
                dist = np.sqrt((self.x_trajectories[i][frame].item() - self.x_trajectories[j][frame].item()) ** 2 +
                               (self.y_trajectories[i][frame].item() - self.y_trajectories[j][frame].item()) ** 2)
                if dist < (2 * Params['car_size']):
                    collision_detected = True
                    collision_x = (self.x_trajectories[i][frame].item() + self.x_trajectories[j][frame].item()) / 2
                    collision_y = (self.y_trajectories[i][frame].item() + self.y_trajectories[j][frame].item()) / 2
                    self.collision_marker.set_data([collision_x], [collision_y])
                    break
            if collision_detected:
                break

        self.collision_marker.set_visible(collision_detected)

        return self.car_images + [self.collision_marker]

    def animate(self, interval=80):  # Default interval is 80 ms
        # Create the animation
        num_frames = len(self.x_trajectories[0])
        self.ani = FuncAnimation(self.fig, self.update, frames=num_frames, blit=True, interval=interval)
        plt.show()

# Example usage (assuming you have the required data loaded in x_trajectories, y_trajectories, x_ends, y_ends)
# animator = TrajectoryAnimator(x_trajectories, y_trajectories, x_ends, y_ends)
# animator.animate()
