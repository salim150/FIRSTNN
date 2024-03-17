import math

class ObjectMovement:
    def __init__(self, x, y, speed, angle):
        self.x = x
        self.y = y
        self.speed = speed
        self.angle = angle

    def move_object(self, delta_speed, delta_angle):
        # Apply constraints on maximum change in speed and angle
        max_delta_speed = 0.5  # Maximum change in speed
        max_delta_angle = math.radians(5)  # Maximum change in angle (in radians)

        # Apply constraints on change in speed and angle
        delta_speed = min(max(delta_speed, -max_delta_speed), max_delta_speed)
        delta_angle = min(max(delta_angle, -max_delta_angle), max_delta_angle)

        # Update speed and angle
        self.speed += delta_speed
        self.angle += delta_angle

        # Update x and y coordinates based on speed and angle
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)

        return self.x, self.y, self.speed, self.angle

'''# Example usage:
initial_x = 0
initial_y = 0
initial_speed = 10
initial_angle = math.radians(45)  # Angle in radians (45 degrees)
obj = ObjectMovement(initial_x, initial_y, initial_speed, initial_angle)

# Change speed and angle
new_x, new_y, new_speed, new_angle = obj.move_object(delta_speed=2, delta_angle=math.radians(20))
print("New coordinates:", new_x, new_y)
print("New speed:", new_speed)
print("New angle (in radians):", new_angle)'''
