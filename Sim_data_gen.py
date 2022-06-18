# ---------- IMPORTS ----------

#Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)

#sklearn imports
from sklearn.model_selection import train_test_split

#Basic dataframe imports
import numpy as np
import pandas as pd
import random
import pickle

#Plotting imports
import matplotlib.pyplot as plt


#  -------------------- COHERENCE MAP AND RESPONSE VARIABLE LOAD AND HANDLING --------------------


data = pd.read_pickle("data/tensor_data_open.pkl")

with open('data/response_var_df.pkl', 'rb') as f:
    response_var_df = pickle.load(f)
response_variables = response_var_df.to_dict('dict')

keys = list()
for key,value in data.items(): #Convert every coherence map into a tensor with key in keys
    data[key] = torch.tensor(value)
    keys.append(key)


#  ---------- IMPLEMENTATION OF TENSOR REGRESSION WITH ADAM OPTIMIZER  ----------

class Candemann_Parafac_module(nn.Module):
    def __init__(self, rank):
        """
        In the constructor we instantiate 71*rank parameters and assign them as
        member parameters.
        """
        super().__init__()
        # Initialize base parameters
        self.rank = rank
        self.beta_0 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
        self.CP = {}



    def forward(self, x):#Forward pass function
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        # HERE WE PREDICT OUR Y VALUE USING CANDEMANN-PARAFAC INFRASTRUCTURE

        finalsum = 0

        for k in range(7):
            for j in range(64):
                for i in range(j+1,64):
                    CPd_sum = 0

                    for d in range(self.rank):

                        CPd_sum += torch.tensor(0.2,dtype=torch.float64) \
                                   * torch.tensor(0.2,dtype=torch.float64) \
                                   * torch.tensor(0.2,dtype=torch.float64)

                    finalsum += x[k][i][j] * CPd_sum

        return self.beta_0 + finalsum

# -------------------- MODEL SIMULATION --------------------

response_variables_sim = {}
test_names = ['MMSE', 'ACE', 'TrailMakingA', 'TrailMakingB', 'DigitSymbol', 'Retention']




for rank_val in range(3):
    rank = rank_val+1
    model_sim = Candemann_Parafac_module(rank=rank)
    for name in test_names:
        response_variables_sim[name+"rank"+str(rank)] = {}
        for key in keys:
            response_variables_sim[name+"rank"+str(rank)][key + "rank"+ str(rank)] = model_sim(data[key])
            print(name  + " for rank = " + str(rank) + " = ongoing")
        print(name + " for rank = " + str(rank) + " = done")

with open('data/Simulated_Data_Tensor_allranks.pkl', 'wb') as handle:
    pickle.dump(response_variables_sim, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open('data/Simulated_Data_Tensor.pkl', 'rb') as f:
#    PPPP = pickle.load(f)