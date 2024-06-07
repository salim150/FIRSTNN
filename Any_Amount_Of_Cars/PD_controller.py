from parameters import Params
import torch

class controller():
     # Initialize the class
    def __init__(self): # Initialize the base clas
        super(controller, self).__init__()  # Retrieve proportional coefficients from Params and assign them to class attributes
        self.K_pd = Params['Prop_coeff_distance'] # Proportional coefficient for distance control
        self.K_pa = Params['Prop_coeff_angle'] # Proportional coefficient for angle control
        self.K_pv = Params['Prop_coeff_speed'] # Proportional coefficient for speed control

# Define the forward method for the controller, which calculates control outputs
    def forward(self, x, y, x_end, y_end, current_speed, current_angle):

        # Calculate the distance between the current position (x, y) and the target position (x_end, y_end)
        distance = torch.sqrt((x-x_end)**2 + (y-y_end)**2)

        # Calculate the angle between the current position and the target position
        angle = torch.arctan2((y_end-y),(x_end-x))
        # Calculate the difference between the desired angle and the current angle
        delta_angle = angle - current_angle

        # Adjust the angle difference to be within [-π, π] radians
        #delta_angle = (delta_angle + torch.pi) % (2 * torch.pi) - torch.pi
        delta_angle = angle - current_angle
        # Adjust delta_angle to be within [-π, π] radians
        if delta_angle > torch.pi:
            delta_angle -= 2 * torch.pi
        elif delta_angle < -torch.pi:
            delta_angle += 2 * torch.pi



        u_speed = self.K_pd * distance - self.K_pv*current_speed # Calculate the control signal for speed
        u_angle = self.K_pa * delta_angle # Calculate the control signal for angle

        return u_speed, u_angle
    





        
        

    

       
