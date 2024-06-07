
import torch
from get_full_traj import trajectory
from parameters import Params


def train_step(model,batched_ic,obstacle,starting_kinematics,criterion, optimizer, device, Length:int, printer=True):

    model.train()
    train_loss = []
    i=0
    batched_ic=batched_ic.reshape(int(batched_ic.shape[0]/Params['batchs size']),Params['batchs size'],2,2)



    # iterate over the batches
    for  sample_batched in batched_ic:
      i+=1
      if (i%100 == 0):print(i)


      loss=0          # initiate loss
      
      pos=sample_batched[:,0,:]
      xf= sample_batched[:,1,:]
      

      input_sample= torch.cat((pos,xf,starting_kinematics, obstacle),1)


      # Perform trajectory
      loss = trajectory(model,criterion,input_sample,loss, Length,device)
      loss =loss/ Length


      # Perform gradient descent
      optimizer.zero_grad()
      loss.sum().backward()
      optimizer.step()
      train_loss.append(loss.cpu().detach().numpy())



    return train_loss