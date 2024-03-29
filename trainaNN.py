#Importing necessary libraries
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from network import NeuralNetwork

#Setting the defined area of where the trajectory can be made
#The max and min will be the defined interval of the x- and y-axis
max_value = 10
min_value = -10

#Input sample
x_start = np.random.uniform(low=min_value, high=max_value)
y_start = np.random.uniform(low=min_value, high=max_value)
x_end = np.random.uniform(low=min_value, high=max_value)
y_end = np.random.uniform(low=min_value, high=max_value)
input_sample = torch.tensor([x_start, y_start, x_end, y_end], dtype=torch.float32)

print(f"The input tensor is the following: ")
print(input_sample)
print("---------------------------------------")
input_size = 4
hidden_size = 20 #Note: The amount of hidden layer can be changed
output_size = 2 #The outputs are the velocity and angular velocity

#Network architecture defined as a class
#Class inherits ffrom nn.Module
class NeuralNetwork(nn.Module):
    def __init__(self): 
        super(NeuralNetwork, self).__init__()
        #Define layers and operations here
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2= nn.Linear(hidden_size, output_size)

    def forward(self, input): 
        #Define forward pass computations here 
        x = input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        print(f"The final output is: {x}")
        print("---------------------------------------")
        return x
        

#Class for calculating loss 
class MSE_Loss(nn.MSELoss):
    def __init__(self):
        super(MSE_Loss, self).__init__()
    def loss_computation(self, prediction, target):
        mse_loss = nn.MSELoss()
        return mse_loss(prediction, target)

#Instantiate model
#model = Neuralnetwork()
model = NeuralNetwork()

#Choose optimizer 
optimizer = Adam(model.parameters(), lr=0.001)

#Instantiate loss function
criterion  = nn.MSELoss()

# Dummy target
target = torch.rand(output_size)

# Training loop
epochs = 1000  # Adjust the number of epochs as needed
for epoch in range(epochs):
    # Forward pass
    output = model(input_sample.unsqueeze(0))  # Unsqueeze to add a batch dimension

    # Calculate loss
    loss = criterion(output, target)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #Importing the output from the class so it can be displayed
    output_x = model.forward(input_sample)

    if epoch % 100 == 0:
        print("---------------------------------------")
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")
        print(f"Output: {output_x}")


print("---------------------------------------")
print(f"The input was: {input_sample}")
print(f"Final output: {output_x}")
print(f"Final loss after optimization and backpropagation: {loss.item()}")
print("---------------------------------------")
print("Training is finished")




