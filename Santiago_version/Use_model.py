import torch
from pathlib import Path
from tester import TEST
from parameters import Params
from get_samples import get_samples
from Network import NeuralNetwork
from plot_trayectory import traj_plot
from get_samples import organize_samples
from plot_trayectory import TrajectoryAnimator

def using_model(Params):

    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # 2. Create model save path
    MODEL_NAME = "Umbumping_cars_V6.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


    device = torch.device("cpu") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Training device: {device}")

    # Instantiate a fresh instance of LinearRegressionModelV2
    loaded_model_1 = NeuralNetwork(Params['Network_layers'])

    # Load model state dict 
    loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # Put model to target device (if your data is on GPU, model will have to be on GPU to make predictions)
    loaded_model_1.to(device)



    print(f"Loaded model:\n{loaded_model_1}")
    print(f"Model on device:\n{next(loaded_model_1.parameters()).device}")
    
    

    train_batchs , test_batchs, obstacle= get_samples(
         Params['obssize'],device,
         Params['#of points'],
         Params['car_size'],
         Params['Environment_limits'])
    starting_kinematics= torch.tensor([0,0]).to(device)

    

    for i in range (15):
      train_batchs , test_batchs, obstacle= get_samples(
         Params['obssize'],device,
         Params['#of points'],
         Params['car_size'],
         Params['Environment_limits'])
      sample_batched=test_batchs[i]
      
      input_sample = torch.tensor([sample_batched[0][0], sample_batched[0][1],
                                   sample_batched[1][0], sample_batched[1][1],
                                   starting_kinematics[0], starting_kinematics[1],
                                   obstacle[0],obstacle[1]]).to(device)
      
      f_traj,_ =TEST(loaded_model_1,input_sample,Params['Length'],device)
      
      traj_plot(f_traj,obstacle,sample_batched[1],i)
      animator = TrajectoryAnimator(obstacle,f_traj,sample_batched[1])
      animator.animate()



if __name__ == "__main__":
    using_model(Params)



