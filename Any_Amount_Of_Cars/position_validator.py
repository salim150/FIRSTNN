import math
from itertools import combinations
from parameters import Params

def positions_okay(x_start, y_start, x_end, y_end):

    # Define the minimum allowed distance between cars to avoid collision
    threshold = 2 * Params['car_size'] + Params['collision_safety']

    # Check distances between starting points
    for (i, j) in combinations(range(Params['number_of_cars']), 2):
        # Calculate the Euclidean distance between two starting points
        start_distance = math.sqrt((x_start[i] - x_start[j]) ** 2 + (y_start[i] - y_start[j]) ** 2)
        if start_distance < threshold: # If the distance is less than the threshold, the positions are invalid
            return False

    # Check distances between ending points
    for (i, j) in combinations(range(Params['number_of_cars']), 2):
        # Calculate the Euclidean distance between two anding points
        end_distance = math.sqrt((x_end[i] - x_end[j]) ** 2 + (y_end[i] - y_end[j]) ** 2)
        if end_distance < threshold:
            return False

    return True