import numpy as np
import pandas as pd
import pickle
import tltorch
import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt
from tensorly.base import tensor_to_vec, partial_tensor_to_vec
from tensorly.datasets.synthetic import gen_image
from tensorly.regression.cp_regression import CPRegressor
import tensorly as tl

# Use tensor regression to reduce size of 3D matrix of size 64x64x7 to single value 64 + 7 = 71

data = pd.read_pickle("data/tensor_data_open.pkl")
# # print shape of the first value of the dictionary
# for key,value in data.items():
#     print(key,value.shape)
# turn each value into a tensor
for key,value in data.items():
    data[key] = torch.tensor(value)

# # view shape of tensor
# print(data["12069"].shape)

# Create a Tensor Regression model that takes in a 64x64x7 tensor and outputs a single vector of size 71
input_shape = (64,64,7)
output_shape = (71,)
batch_size = 32
x = data["12069"]

# from tensorly.regression.cp_regression import CPRegressor
# model = CPRegressor(input_shape, output_shape, batch_size)

# from tensorly.regression.tucker_regression import TuckerRegressor
# model = TuckerRegressor(input_shape, output_shape, batch_size)

# Hvad skal weight ranks være? Skal vi bruge CP Regression eller Tucker Regression?




# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # set x to device
# x = x.to(device)
# print(x.shape)