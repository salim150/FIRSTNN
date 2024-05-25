import torch


Params = {
    'Network_layers' : [6, 64,64,64, 2],
    'Length' : 30,
    'batchs' : 100,
    'points_per_batch' : 10,
    'radius' : 0.5,
    'Environment_limits' : torch.tensor([[-10,10],[-10,10]]),
    'epochs' : 50000,
    'Learning_rate' : 1e-3,
    'max_speed' : 3,
    'max_omega' : 1,
    'max_acc' : 0.2,
    'max_ang_acc' : 0.4,
    'dt' : 0.075,

    'collision_safety' : 0.1,
    'car_size' : 0.6,
  
    'start_radius' : 0,

    'Prop_coeff_distance' : 0.5,
    'Prop_coeff_angle' : 0.7,
    'Prop_coeff_speed' : 0.5,
    'nn_coeff_speed' : 6,
    'nn_coeff_angle' : 8,

    'alpha' : 1,
    'beta' : 2,
    'gamma' : 3,
    'delta' : 5,
    'obssize' : 3,
    'high_value' : 100,

    'number_of_cars' : 15,
    'trajectory_length' : 40
}