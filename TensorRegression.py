
# ---------- IMPORTS ----------

#Torch imports
import torch
import torch as nn
import torch.optim as optim
torch.manual_seed(0)

#sklearn imports
from sklearn.model_selection import train_test_split

#Basic dataframe imports
import numpy as np
import pandas as pd
import pickle

#Plotting imports
import matplotlib.pyplot as plt


#  ---------- COHERENCE AND RESPONSE VARIABLE IMPORT AND HANDLING ----------
data = pd.read_pickle("data/tensor_data_open.pkl")

with open('data/response_var_df.pkl', 'rb') as f:
    response_var_df = pickle.load(f)
response_variables = response_var_df.to_dict('dict')


"""
keys = list()
for key,value in data.items(): #Convert every coherence map into a tensor with key in keys
    data[key] = torch.tensor(value)
    keys.append(key)
"""

keys = list()
for key,value in data.items(): #Convert every coherence map into a tensor with key in keys
    data[key] = torch.tensor(value)
    keys.append(key)




#  ---------- DEFINING OUR ADAM OPTIMIZER  ----------

class Regress_Loss(torch.nn.Module):

    def __init__(self):
        super(Regress_Loss, self).__init__()

    def forward(self, x, y):
        y_shape = y.size()[1]
        x_added_dim = x.unsqueeze(1)
        x_stacked_along_dimension1 = x_added_dim.repeat(1, NUM_WORDS, 1)
        diff = torch.sum((y - x_stacked_along_dimension1) ** 2, 2)
        totloss = torch.sum(torch.sum(torch.sum(diff)))
        return totloss


#  ---------- IMPLEMENTATION OF TENSOR REGRESSION  ----------
# --- CP function ---
def CPdict(rank):
    CP = {}
    for i in range(rank):
        CP["gamma"+str(i)] = {}
        CP["alpha"+str(i)] = {}
        for j in range(7):
            CP["gamma" + str(i)]["g"+str(i)+str(j)] = 1
        for k in range(64):
            CP["alpha" + str(i)]["a" + str(i) + str(k)] = 1

        return CP

CPs = CPdict(rank=3)

# --- Cost function ---

def Cost_Function(Xn, Yn, rank, CP_dict):

    finalsum = 0

    for k in range(7):
        for j in range(64):
            for i in range(64):
                if i>j:
                    CPd_sum=0

                    for d in range(rank):
                        CPd_sum += CP["alpha" + str(d)][i]

                        CPd_sum+=CP["alpha"+str(d)][i]*CP["alpha"+str(d)][i]*CP_dict["gamma"+str(d)][j]

                    finalsum += Xn[i][j][k]*CPd_sum

    return (Yn - beta_0 + finalsum)**2

f = 2

#print(Cost_Function(Xn = x, Yn = 29, rank=2, CP=CP))

# --- Adam optimization ---


"""
if __name__ == '__main__':

    torch.manual_seed(42)
    X, X_test, y, y_test = Load_data("data/PCA+Y-OPEN.pkl")
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

"""
#  ---------- CURRENTLY UNUSED CODE  ----------




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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set x to device
x = x.to(device)
print(x.shape)

"""