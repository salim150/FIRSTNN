import torch

# Définition des constantes
alpha = 1
beta = 10
gamma = 20

outside_penalty_value = 500  # Valeur de la pénalité pour sortir du terrain
obstacle_penalty_value = 1000  # Valeur de la pénalité pour toucher un obstacle


# Coordonnées du point d'arrivée et de l'obstacle
x_goal, y_goal = 10, 10
#x_obstacle, y_obstacle = 5, 5
obstacle_size = 3

# Limites de la zone
x_min, x_max = 0, 20
y_min, y_max = 0, 20

high_value = 10000000  # éviter asymptote de la fonction

x_robot, y_robot = 1, 1  # Coordonnées au hasard

# Loss Function
def loss_function(x_robot, y_robot, x_obstacle, y_obstacle):

    # Distance par rapport au point d'arrivée
    distance_to_goal = (x_robot - x_goal) ** 2 + (y_robot - y_goal) ** 2

    # Pénalité pour sortir de la zone
    terrain_penalty = torch.min(high_value, -torch.log(1 - torch.exp(
        -outside_penalty_value * (torch.min(x_robot - x_min, x_max - x_robot, y_robot - y_min,
                                            y_max - y_robot)))))

    # Pénalité pour toucher un obstacle
    obstacle_penalty = torch.min(high_value, -torch.log(1 - torch.exp(
        -obstacle_penalty_value * ((x_robot - x_obstacle) ** 2 + (y_robot - y_obstacle) ** 2 -
                                   obstacle_size ** 2))))

    # Calcul loss
    loss = alpha * distance_to_goal + beta * terrain_penalty + gamma * obstacle_penalty

    return loss


# Example usage
loss = loss_function(x_robot, y_robot)
print(loss)
