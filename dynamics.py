import numpy as np

class ObjectMovement:
    def __init__(self, x, y, speed, angle):
        self.x = x
        self.y = y
        self.speed = speed
        self.angle = angle

    def move_object(self, delta_speed, delta_angle):
        # Apply constraints on maximum change in speed and angle
        max_delta_speed = 0.2  # Maximum change in speed
        max_speed = 1
        max_delta_angle = np.radians(20)  # Maximum change in angle (in radians)

        # Apply constraints on change in speed and angle
        delta_speed = min(max(delta_speed, -max_delta_speed), max_delta_speed)
        delta_angle = min(max(delta_angle, -max_delta_angle), max_delta_angle)

        # Update speed and angle
        self.speed = min(max(self.speed + delta_speed, 0), max_speed)
        self.angle += delta_angle

        # Update x and y coordinates based on speed and angle
        self.x += self.speed * np.cos(self.angle)
        self.y += self.speed * np.sin(self.angle)

        return self.x, self.y, self.speed, self.angle

'''# Example usage:
initial_x = 0
initial_y = 0
initial_speed = 0.3
initial_angle = np.radians(45)  # Angle in radians (45 degrees)
obj = ObjectMovement(initial_x, initial_y, initial_speed, initial_angle)

# Change speed and angle
new_x, new_y, new_speed, new_angle = obj.move_object(delta_speed=2, delta_angle=np.radians(20))
print("New coordinates:", new_x, new_y)
print("New speed:", new_speed)
print("New angle (in radians):", new_angle)'''
