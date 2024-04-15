#Obstacle generator - Should be imported into the main file to see how well the network can avoid obstacles

#Import dependencies
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import numpy as np

#Can and should be changed to fit wihtin the boundaries of the plot
low_limit = -6
upper_limit = 0

#Class to generate obstacles 
class Obstacle_generator():
    def __init__(self, min_coord = low_limit, max_coord=upper_limit):
        self.min_coord = min_coord
        self.max_coord = max_coord
    
    # Want to randomly generate the coordinates for the obstacles
    def generate_obstacle(self):
        xo1 = np.random.uniform(low=self.min_coord, high=self.max_coord)
        xo2 = xo1 + np.random.uniform(low=self.min_coord, high=self.max_coord)
        yo1 = np.random.uniform(low=self.min_coord, high=self.max_coord)
        yo2 = yo1 + np.random.uniform(low=self.min_coord, high=self.max_coord)

        vertices = [(xo1, yo1), (xo1, yo2), (xo2, yo2), (xo2, yo1), (xo1, yo1)]  # Closing the path

        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]

        path = Path(vertices, codes)

        pathpatch = PathPatch(path, facecolor='grey', edgecolor='black')  # Change facecolor to the desired fill color
        #Choose colour differently from the colour of simulated trajectory 

        return pathpatch


class Obstacle_Avoidance: 
    def __init__(self, obstacle):
        self.obstacle = obstacle
    #Come up with a way of how the vehicle should move when it is getting to close the obstacle
    #How will the car dynamics be implemented to avoid the obstacle
    def avoid_obstacle(self, x, y):
        # Here you can implement your obstacle avoidance logic
        # For example, you can modify the trajectory to avoid collision with the obstacle
        if (x >= self.obstacle[0] + 0.5 and x <= self.obstacle[2] - 0.5) and (y >= self.obstacle[1] + 0.5 and y <= self.obstacle[3] - 0.5):
            # If the current position is close to obstacle move away from it 
            x += np.random.uniform(-1, 1)
            y += np.random.uniform(-1, 1)
        return x, y
 



