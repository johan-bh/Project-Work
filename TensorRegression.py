import numpy as np
import pandas as pd
import pickle
import tltorch
import torch
from torch import nn
import numpy as np

# Use tensor regression to reduce size of 3D matrix of size 64x64x7 to single value 64 + 7 = 71

data = pd.read_pickle("data/tensor_data_open.pkl")
# # print shape of the first value of the dictionary
# for key,value in data.items():
#     print(key,value.shape)
# turn each value into a tensor
keys = list()
for key,value in data.items():
    data[key] = torch.tensor(value)
    keys.append(key)

# # view shape of tensor
# print(data["12069"].shape)

# Create a Tensor Regression model that takes in a 64x64x7 tensor and outputs a single vector of size 71
input_shape = (64,64,7)
output_shape = (71,)
batch_size = 32
x = data["12069"]

print(x)

from tensorly.decomposition import SymmetricCP
model = SymmetricCP(input_shape, output_shape, batch_size)
print(data, )





# Hvad skal weight ranks være?




# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # set x to device
# x = x.to(device)
# print(x.shape)