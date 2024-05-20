import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt





def get_samples(obsize,device,clouds=5,radius=1,limits = torch.tensor([[-10,10],[-10,10]])):

  points=torch.tensor([[],[]])
  obstacle=torch.transpose((torch.rand(2,1)*limits.diff()+limits[:,0].unsqueeze(dim=1)),0,1).squeeze()
  limits = limits +radius* torch.tensor([[1,-1],[1,-1]])
  while points.shape[1] < 2*clouds:
        # Generate random points within the rectangle using torch
        centers= torch.rand(2,2*clouds)*limits.diff()+limits[:,0].unsqueeze(dim=1)

        # Calculate distances from the hole center
        distances = torch.sqrt((centers[0] - obstacle[0])**2 + (centers[1] - obstacle[1])**2)

        # Filter out points that are inside the hole
        mask = distances > (obsize+radius)
        valid_points = centers[:,mask]

        points=torch.cat((points,valid_points),1)
  points= torch.transpose(points[:,:2*clouds],0,1).unsqueeze(1).reshape(clouds,2,2)


  train_batchs = points[0:int(0.8*clouds),:,:].to(device)
  test_batchs= points[int(0.8*clouds):clouds,:,:].to(device)
  return train_batchs,test_batchs, obstacle

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