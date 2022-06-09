#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 18:27:57 2022

@author: smilladue

Inspiration / general build up and understanding of pytorhc: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
How to change from classification to regression problem: https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md
"""

import os
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def Load_data(pkl_file):
    A = pd.read_pickle(pkl_file)
    X_train, X_test, y_train, y_test = train_test_split(
       A.iloc[:, :-4], A.iloc[:, -4:], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

class DataSet(torch.utils.data.Dataset):
  '''
  Prepare the Boston dataset for regression
  '''

  def __init__(self, X, y, scale_data=True):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if necessary
      if scale_data:
          X = StandardScaler().fit_transform(X)
      self.X = torch.from_numpy(X)
      self.y = torch.from_numpy(y)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]

class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(50, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 4)
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)

#Load_data("data/PCA+Y-OPEN.pkl")

if __name__ == '__main__':

    torch.manual_seed(42)
    X, X_test, y, y_test = Load_data("data/PCA+Y-OPEN.pkl")
    Xn=X.to_numpy()
    yn=y.to_numpy()
    dataset = DataSet(Xn, yn)
    #dataset = X.to_numpy(), y.to_numpy()[0]
    trainloader = torch.utils.data.DataLoader(dataset)
    mlp = MLP()
    
    # Define the loss function (mean absolute error between input x and target y) and optimizer
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    count=0
      
    # Run the training loop
    for epoch in range(0, 5): # 5 epochs at maximum
      # Print epoch
      print(f'Starting epoch {epoch+1}')
      
      # Set current loss value
      current_loss = 0.0
      
      # Iterate over the DataLoader for training data
      for i, data in enumerate(trainloader, 0):

        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        #targets = targets[0].reshape((targets.shape[0], 1))
      
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        outputs = mlp(inputs)
        
        # Compute loss
        loss = loss_function(outputs, targets)

       
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        
        # Print statistics
        current_loss += loss.item()
        if i % 10 == 0:
            count +=1
            print('Loss after mini-batch %5d: %.3f' %
                  (i + 1, current_loss / 155))
            current_loss = 0.0
            
      
    # Process is complete.
    print('Training process has finished.')
    



"""
        targ1, targ2, targ3, targ4,  = targets["MMSE"], targets["ACE"], targets["TrailMakingA"], targets["TrailMakingB"]
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        out1, out2, out3, out4 = mlp(inputs)
        loss1 = loss_function(out1, targ1)
        loss2 = loss_function(out2, targ2)
        loss3 = loss_function(out3, targ3)
        loss4 = loss_function(out4, targ4)
        loss = loss1 + loss2 + loss3 + loss4 


class mlp(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    
model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
"""