import numpy as np
import pandas as pd
import pickle
import tltorch
import torch
from torch import nn
import numpy as np
import tensorly as tl

import matplotlib.pyplot as plt
# Use tensor regression to reduce size of 3D matrix of size 64x64x7 to single value 64 + 7 = 71

data = pd.read_pickle("data/tensor_data_open.pkl")

#with open('data/response_var_df.pkl', 'rb') as f:
#    response_var_df = pickle.load(f)



keys = list()
for key,value in data.items(): #Convert every coherence map into a tensor with key in keys
    data[key] = torch.tensor(value)
    keys.append(key)

# Create a Tensor Regression model that takes in a 64x64x7 tensor and outputs a single vector of size 71
input_shape = (64,64,7)
batch_size = 32
x = data["12069"]

#Actual Implementation


def Loss_function(Xn, Yn, rank=1):
    gamma = np.ones((7, 1))
    alpha = np.ones((64, 1))
    CP = {}
    for i in range(rank):
        CP["gamma" + str(i)] = gamma
        CP["alpha" + str(i)] = alpha

    for k in range(7):
        for j in range(64):
            for i in range(64):
                if i<=j:
                    continue

                for d in range(rank):
                    CP["alpha"+str(d)][i]*CP["alpha"+str(d)][i]*CP["beta"+str(d)][j]

                Xn[i][j][k]

Loss_function(x, 29, rank=2)




"""
if __name__ == '__main__':

    torch.manual_seed(42)
    X, X_test, y, y_test = Load_data("data/tensor_data_open.pkl")
    Xn = X.to_numpy()
    yn = y.to_numpy()
    dataset = DataSet(Xn, yn)
    # dataset = X.to_numpy(), y.to_numpy()[0]
    trainloader = torch.utils.data.DataLoader(dataset)
    mlp = MLP()

    # Define the loss function (mean absolute error between input x and target y) and optimizer
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    count = 0

    # Run the training loop
    for epoch in range(0, 5):  # 5 epochs at maximum
        # Print epoch
        print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):

            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            # targets = targets[0].reshape((targets.shape[0], 1))

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
                count += 1
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / 155))
                current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')

#Function for plotting a 2d tensor

def showTensor(aTensor):
    plt.figure()
    plt.imshow(aTensor.numpy())
    plt.colorbar()
    plt.gca().invert_xaxis()
    plt.show()



#experimentation

from tensorly.decomposition import parafac
factors = parafac(x,rank=2)
print(factors.shape)
new_x = tl.cp_to_tensor(factors)

showTensor(x[:,:,1])
"""

#Tensor Regression






# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # set x to device
# x = x.to(device)
# print(x.shape)
