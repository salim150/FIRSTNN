import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def get_samples(clouds=5,size=20,radius=1,limits = torch.tensor([[-10,10],[-10,10]])):

  centers= torch.rand(2,clouds)*limits.diff()+limits[:,0].unsqueeze(dim=1)



  angles=torch.rand(size)
  positions=centers.T.reshape(clouds,2,1) +  torch.rand(size)*torch.tensor([[radius],[radius]])*torch.cat((torch.cos(angles).unsqueeze(0),torch.sin(angles).unsqueeze(0)),0)
  #visualise data:
  for i in range(min(10,clouds)):
    plt.plot(positions[i,0,:],positions[i,1,:],'o',markersize=2)
  return positions

def organize_samples(Params,possible_points,device):

  #organize the points into starting , ending points and divide them into test & train.
  #x0 = torch.transpose(possible_points[0,:,0].unsqueeze(0),0,1)
  a=torch.randint( Params['points_per_cloud'] , (1,Params['#of points'])) #this two arrays are just some indexing shuffleing to separete themn the test and train data.
  b=torch.randperm(Params['#of points'])

  train_batchs_i=torch.transpose(possible_points[b.squeeze(0)[0:int(0.8*Params['#of points']-1)],:,a.squeeze(0)[:int(0.8*Params['#of points']-1)]].unsqueeze(0),0,1).squeeze(1)
  test_batchs_i=torch.transpose(possible_points[b.squeeze(0)[int(0.8*Params['#of points']-1):Params['#of points']],:,a.squeeze(0)[int(0.8*Params['#of points']-1):Params['#of points']]].unsqueeze(0),0,1).squeeze(1)

  #xf = torch.tensor([[0],[0]])

  a=torch.randint( Params['points_per_cloud'] , (1,Params['#of points'])) #this two arrays are just some indexing shuffleing to separete themn the test and train data.
  b=torch.randperm(Params['#of points'])

  train_batchs_f=torch.transpose(possible_points[b.squeeze(0)[0:int(0.8*Params['#of points']-1)],:,a.squeeze(0)[:int(0.8*Params['#of points']-1)]].unsqueeze(0),0,1).squeeze(1)
  test_batchs_f=torch.transpose(possible_points[b.squeeze(0)[int(0.8*Params['#of points']-1):Params['#of points']],:,a.squeeze(0)[int(0.8*Params['#of points']-1):Params['#of points']]].unsqueeze(0),0,1).squeeze(1)

  train_batchs_f=torch.ones_like(train_batchs_i)*5
  #train_batchs_i=torch.ones_like(train_batchs_f)*-5
  test_batchs_f=torch.ones_like(test_batchs_i)*5
  #test_batchs_i=torch.ones_like(test_batchs_f)*-5
  



  train_batchs = torch.stack((train_batchs_i, train_batchs_f),1).to(device)
  test_batchs= torch.stack((test_batchs_i, test_batchs_f),1)

  return train_batchs,test_batchs