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

        # Here we define our torch parameters, which is our CP variables.
        # I ended up having to do this as each variable on at a time
        # because i couldn't find a way to set a string in as the name for a variable,
        # and torch didn't accept dictionaries.

        if self.rank > 0:
            self.g00 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g01 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g02 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g03 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g04 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g05 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g06 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))

            self.a00 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a01 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a02 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a03 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a04 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a05 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a06 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a07 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a08 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a09 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a010 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a011 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a012 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a013 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a014 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a015 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a016 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a017 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a018 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a019 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a020 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a021 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a022 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a023 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a024 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a025 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a026 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a027 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a028 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a029 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a030 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a031 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a032 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a033 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a034 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a035 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a036 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a037 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a038 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a039 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a040 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a041 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a042 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a043 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a044 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a045 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a046 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a047 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a048 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a049 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a050 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a051 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a052 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a053 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a054 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a055 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a056 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a057 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a058 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a059 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a060 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a061 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a062 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a063 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))

            self.CP["gamma0"] = {}

            self.CP["gamma0"]["g00"] = self.g00
            self.CP["gamma0"]["g01"] = self.g01
            self.CP["gamma0"]["g02"] = self.g02
            self.CP["gamma0"]["g03"] = self.g03
            self.CP["gamma0"]["g04"] = self.g04
            self.CP["gamma0"]["g05"] = self.g05
            self.CP["gamma0"]["g06"] = self.g06

            self.CP["alpha0"] = {}

            self.CP["alpha0"]["a00"] = self.a00
            self.CP["alpha0"]["a01"] = self.a01
            self.CP["alpha0"]["a02"] = self.a02
            self.CP["alpha0"]["a03"] = self.a03
            self.CP["alpha0"]["a04"] = self.a04
            self.CP["alpha0"]["a05"] = self.a05
            self.CP["alpha0"]["a06"] = self.a06
            self.CP["alpha0"]["a07"] = self.a07
            self.CP["alpha0"]["a08"] = self.a08
            self.CP["alpha0"]["a09"] = self.a09
            self.CP["alpha0"]["a010"] = self.a010
            self.CP["alpha0"]["a011"] = self.a011
            self.CP["alpha0"]["a012"] = self.a012
            self.CP["alpha0"]["a013"] = self.a013
            self.CP["alpha0"]["a014"] = self.a014
            self.CP["alpha0"]["a015"] = self.a015
            self.CP["alpha0"]["a016"] = self.a016
            self.CP["alpha0"]["a017"] = self.a017
            self.CP["alpha0"]["a018"] = self.a018
            self.CP["alpha0"]["a019"] = self.a019
            self.CP["alpha0"]["a020"] = self.a020
            self.CP["alpha0"]["a021"] = self.a021
            self.CP["alpha0"]["a022"] = self.a022
            self.CP["alpha0"]["a023"] = self.a023
            self.CP["alpha0"]["a024"] = self.a024
            self.CP["alpha0"]["a025"] = self.a025
            self.CP["alpha0"]["a026"] = self.a026
            self.CP["alpha0"]["a027"] = self.a027
            self.CP["alpha0"]["a028"] = self.a028
            self.CP["alpha0"]["a029"] = self.a029
            self.CP["alpha0"]["a030"] = self.a030
            self.CP["alpha0"]["a031"] = self.a031
            self.CP["alpha0"]["a032"] = self.a032
            self.CP["alpha0"]["a033"] = self.a033
            self.CP["alpha0"]["a034"] = self.a034
            self.CP["alpha0"]["a035"] = self.a035
            self.CP["alpha0"]["a036"] = self.a036
            self.CP["alpha0"]["a037"] = self.a037
            self.CP["alpha0"]["a038"] = self.a038
            self.CP["alpha0"]["a039"] = self.a039
            self.CP["alpha0"]["a040"] = self.a040
            self.CP["alpha0"]["a041"] = self.a041
            self.CP["alpha0"]["a042"] = self.a042
            self.CP["alpha0"]["a043"] = self.a043
            self.CP["alpha0"]["a044"] = self.a044
            self.CP["alpha0"]["a045"] = self.a045
            self.CP["alpha0"]["a046"] = self.a046
            self.CP["alpha0"]["a047"] = self.a047
            self.CP["alpha0"]["a048"] = self.a048
            self.CP["alpha0"]["a049"] = self.a049
            self.CP["alpha0"]["a050"] = self.a050
            self.CP["alpha0"]["a051"] = self.a051
            self.CP["alpha0"]["a052"] = self.a052
            self.CP["alpha0"]["a053"] = self.a053
            self.CP["alpha0"]["a054"] = self.a054
            self.CP["alpha0"]["a055"] = self.a055
            self.CP["alpha0"]["a056"] = self.a056
            self.CP["alpha0"]["a057"] = self.a057
            self.CP["alpha0"]["a058"] = self.a058
            self.CP["alpha0"]["a059"] = self.a059
            self.CP["alpha0"]["a060"] = self.a060
            self.CP["alpha0"]["a061"] = self.a061
            self.CP["alpha0"]["a062"] = self.a062
            self.CP["alpha0"]["a063"] = self.a063

        if rank > 1:
            self.g10 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g11 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g12 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g13 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g14 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g15 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g16 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))

            self.a10 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a11 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a12 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a13 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a14 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a15 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a16 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a17 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a18 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a19 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a110 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a111 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a112 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a113 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a114 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a115 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a116 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a117 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a118 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a119 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a120 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a121 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a122 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a123 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a124 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a125 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a126 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a127 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a128 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a129 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a130 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a131 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a132 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a133 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a134 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a135 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a136 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a137 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a138 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a139 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a140 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a141 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a142 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a143 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a144 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a145 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a146 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a147 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a148 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a149 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a150 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a151 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a152 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a153 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a154 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a155 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a156 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a157 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a158 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a159 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a160 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a161 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a162 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a163 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))

            self.CP["gamma1"] = {}

            self.CP["gamma1"]["g10"] = self.g10
            self.CP["gamma1"]["g11"] = self.g11
            self.CP["gamma1"]["g12"] = self.g12
            self.CP["gamma1"]["g13"] = self.g13
            self.CP["gamma1"]["g14"] = self.g14
            self.CP["gamma1"]["g15"] = self.g15
            self.CP["gamma1"]["g16"] = self.g16

            self.CP["alpha1"] = {}

            self.CP["alpha1"]["a10"] = self.a10
            self.CP["alpha1"]["a11"] = self.a11
            self.CP["alpha1"]["a12"] = self.a12
            self.CP["alpha1"]["a13"] = self.a13
            self.CP["alpha1"]["a14"] = self.a14
            self.CP["alpha1"]["a15"] = self.a15
            self.CP["alpha1"]["a16"] = self.a16
            self.CP["alpha1"]["a17"] = self.a17
            self.CP["alpha1"]["a18"] = self.a18
            self.CP["alpha1"]["a19"] = self.a19
            self.CP["alpha1"]["a110"] = self.a110
            self.CP["alpha1"]["a111"] = self.a111
            self.CP["alpha1"]["a112"] = self.a112
            self.CP["alpha1"]["a113"] = self.a113
            self.CP["alpha1"]["a114"] = self.a114
            self.CP["alpha1"]["a115"] = self.a115
            self.CP["alpha1"]["a116"] = self.a116
            self.CP["alpha1"]["a117"] = self.a117
            self.CP["alpha1"]["a118"] = self.a118
            self.CP["alpha1"]["a119"] = self.a119
            self.CP["alpha1"]["a120"] = self.a120
            self.CP["alpha1"]["a121"] = self.a121
            self.CP["alpha1"]["a122"] = self.a122
            self.CP["alpha1"]["a123"] = self.a123
            self.CP["alpha1"]["a124"] = self.a124
            self.CP["alpha1"]["a125"] = self.a125
            self.CP["alpha1"]["a126"] = self.a126
            self.CP["alpha1"]["a127"] = self.a127
            self.CP["alpha1"]["a128"] = self.a128
            self.CP["alpha1"]["a129"] = self.a129
            self.CP["alpha1"]["a130"] = self.a130
            self.CP["alpha1"]["a131"] = self.a131
            self.CP["alpha1"]["a132"] = self.a132
            self.CP["alpha1"]["a133"] = self.a133
            self.CP["alpha1"]["a134"] = self.a134
            self.CP["alpha1"]["a135"] = self.a135
            self.CP["alpha1"]["a136"] = self.a136
            self.CP["alpha1"]["a137"] = self.a137
            self.CP["alpha1"]["a138"] = self.a138
            self.CP["alpha1"]["a139"] = self.a139
            self.CP["alpha1"]["a140"] = self.a140
            self.CP["alpha1"]["a141"] = self.a141
            self.CP["alpha1"]["a142"] = self.a142
            self.CP["alpha1"]["a143"] = self.a143
            self.CP["alpha1"]["a144"] = self.a144
            self.CP["alpha1"]["a145"] = self.a145
            self.CP["alpha1"]["a146"] = self.a146
            self.CP["alpha1"]["a147"] = self.a147
            self.CP["alpha1"]["a148"] = self.a148
            self.CP["alpha1"]["a149"] = self.a149
            self.CP["alpha1"]["a150"] = self.a150
            self.CP["alpha1"]["a151"] = self.a151
            self.CP["alpha1"]["a152"] = self.a152
            self.CP["alpha1"]["a153"] = self.a153
            self.CP["alpha1"]["a154"] = self.a154
            self.CP["alpha1"]["a155"] = self.a155
            self.CP["alpha1"]["a156"] = self.a156
            self.CP["alpha1"]["a157"] = self.a157
            self.CP["alpha1"]["a158"] = self.a158
            self.CP["alpha1"]["a159"] = self.a159
            self.CP["alpha1"]["a160"] = self.a160
            self.CP["alpha1"]["a161"] = self.a161
            self.CP["alpha1"]["a162"] = self.a162
            self.CP["alpha1"]["a163"] = self.a163

        if rank > 2:
            self.g20 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g21 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g22 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g23 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g24 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g25 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.g26 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))

            self.a20 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a21 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a22 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a23 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a24 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a25 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a26 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a27 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a28 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a29 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a210 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a211 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a212 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a213 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a214 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a215 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a216 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a217 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a218 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a219 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a220 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a221 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a222 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a223 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a224 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a225 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a226 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a227 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a228 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a229 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a230 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a231 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a232 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a233 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a234 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a235 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a236 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a237 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a238 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a239 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a240 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a241 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a242 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a243 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a244 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a245 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a246 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a247 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a248 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a249 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a250 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a251 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a252 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a253 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a254 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a255 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a256 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a257 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a258 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a259 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a260 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a261 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a262 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))
            self.a263 = torch.nn.Parameter(torch.tensor(0.2,dtype=torch.float64))

            self.CP["gamma2"] = {}

            self.CP["gamma2"]["g20"] = self.g20
            self.CP["gamma2"]["g21"] = self.g21
            self.CP["gamma2"]["g22"] = self.g22
            self.CP["gamma2"]["g23"] = self.g23
            self.CP["gamma2"]["g24"] = self.g24
            self.CP["gamma2"]["g25"] = self.g25
            self.CP["gamma2"]["g26"] = self.g26

            self.CP["alpha2"] = {}

            self.CP["alpha2"]["a20"] = self.a20
            self.CP["alpha2"]["a21"] = self.a21
            self.CP["alpha2"]["a22"] = self.a22
            self.CP["alpha2"]["a23"] = self.a23
            self.CP["alpha2"]["a24"] = self.a24
            self.CP["alpha2"]["a25"] = self.a25
            self.CP["alpha2"]["a26"] = self.a26
            self.CP["alpha2"]["a27"] = self.a27
            self.CP["alpha2"]["a28"] = self.a28
            self.CP["alpha2"]["a29"] = self.a29
            self.CP["alpha2"]["a210"] = self.a210
            self.CP["alpha2"]["a211"] = self.a211
            self.CP["alpha2"]["a212"] = self.a212
            self.CP["alpha2"]["a213"] = self.a213
            self.CP["alpha2"]["a214"] = self.a214
            self.CP["alpha2"]["a215"] = self.a215
            self.CP["alpha2"]["a216"] = self.a216
            self.CP["alpha2"]["a217"] = self.a217
            self.CP["alpha2"]["a218"] = self.a218
            self.CP["alpha2"]["a219"] = self.a219
            self.CP["alpha2"]["a220"] = self.a220
            self.CP["alpha2"]["a221"] = self.a221
            self.CP["alpha2"]["a222"] = self.a222
            self.CP["alpha2"]["a223"] = self.a223
            self.CP["alpha2"]["a224"] = self.a224
            self.CP["alpha2"]["a225"] = self.a225
            self.CP["alpha2"]["a226"] = self.a226
            self.CP["alpha2"]["a227"] = self.a227
            self.CP["alpha2"]["a228"] = self.a228
            self.CP["alpha2"]["a229"] = self.a229
            self.CP["alpha2"]["a230"] = self.a230
            self.CP["alpha2"]["a231"] = self.a231
            self.CP["alpha2"]["a232"] = self.a232
            self.CP["alpha2"]["a233"] = self.a233
            self.CP["alpha2"]["a234"] = self.a234
            self.CP["alpha2"]["a235"] = self.a235
            self.CP["alpha2"]["a236"] = self.a236
            self.CP["alpha2"]["a237"] = self.a237
            self.CP["alpha2"]["a238"] = self.a238
            self.CP["alpha2"]["a239"] = self.a239
            self.CP["alpha2"]["a240"] = self.a240
            self.CP["alpha2"]["a241"] = self.a241
            self.CP["alpha2"]["a242"] = self.a242
            self.CP["alpha2"]["a243"] = self.a243
            self.CP["alpha2"]["a244"] = self.a244
            self.CP["alpha2"]["a245"] = self.a245
            self.CP["alpha2"]["a246"] = self.a246
            self.CP["alpha2"]["a247"] = self.a247
            self.CP["alpha2"]["a248"] = self.a248
            self.CP["alpha2"]["a249"] = self.a249
            self.CP["alpha2"]["a250"] = self.a250
            self.CP["alpha2"]["a251"] = self.a251
            self.CP["alpha2"]["a252"] = self.a252
            self.CP["alpha2"]["a253"] = self.a253
            self.CP["alpha2"]["a254"] = self.a254
            self.CP["alpha2"]["a255"] = self.a255
            self.CP["alpha2"]["a256"] = self.a256
            self.CP["alpha2"]["a257"] = self.a257
            self.CP["alpha2"]["a258"] = self.a258
            self.CP["alpha2"]["a259"] = self.a259
            self.CP["alpha2"]["a260"] = self.a260
            self.CP["alpha2"]["a261"] = self.a261
            self.CP["alpha2"]["a262"] = self.a262
            self.CP["alpha2"]["a263"] = self.a263

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

                        CPd_sum += self.CP["alpha" + str(d)]["a"+str(d)+str(i)] \
                                   * self.CP["alpha" + str(d)]["a"+str(d)+str(j)] \
                                   * self.CP["gamma" + str(d)]["g"+str(d)+str(k)]

                    finalsum += x[k][i][j] * CPd_sum

        return self.beta_0 + finalsum

# ----- Getting R-squared -----


def R_squared(test_preds, true_test, true_train):
    #(test_preds, true_test, true_values_train)
    #mean_pred = sum(y_pred_vals)/len(y_pred_vals)
    mean_true = sum(true_train) / len(true_train)


    MSE_top = 0
    MSE_bottom = 0

    for i,j in zip(true_test,test_preds):
        MSE_top += (i-j)**2
    for i in true_test:
        MSE_bottom += (i-mean_true)**2

    R2 = 1-(MSE_top)/(MSE_bottom)

    return R2, MSE_top, MSE_bottom

# -------------------- MODEL SIMULATION --------------------

response_variables_sim = {}
test_names = ['MMSE']#, 'ACE', 'TrailMakingA', 'TrailMakingB', 'DigitSymbol', 'Retention']


for rank_val in range(3):
    rank = rank_val+1
    model_sim = Candemann_Parafac_module(rank=rank)
    for name in test_names:
        response_variables_sim[name] = {}
        for key in keys:
            response_variables_sim[name][key + "rank"+ str(rank)] = model_sim(data[key])
        print(name  + "for rank = " + str(rank) + " = done")




# --------------------MODEL TRAINING --------------------


random.seed(420)

for test in test_names:#Looping over all tests seperately

    for rank_iter in range(3):
        rank = rank_iter+1
        torch.manual_seed(20)


        model = Candemann_Parafac_module(rank=rank)

        learning_rate = 0.05
        n_iters = len(data)
        train_stop = 250

        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_preds = []
        test_preds = []
        true_values_train = []
        true_values_test = []



        for epoch in range(n_iters):  # looping over all values in dataset for given rank and test
            print(test + " : " + "rank = " + str(rank) + " : " + str(epoch))
            # predict = forward pass with our model
            try:
                y_true = response_variables[test][keys[epoch]+ "rank"+ str(rank)]
            except:
                continue

            y_predicted = model(data[keys[epoch]])

            y_true = response_variables[test][keys[epoch]]

            # loss

            l = loss(torch.tensor(y_true, dtype=torch.float64), y_predicted)


            # Values for R squared
            if epoch < train_stop:
                train_preds.append(y_predicted)
                true_values_train.append(y_true)
            else:
                test_preds.append(y_predicted)
                true_values_test.append(y_true)
                continue

            # Backward MSE and step

            print("We predict" + str(y_predicted) + " : Real value " + str(y_true))
            # calculate gradients = backward pass
            l.backward()

            # update weights
            optimizer.step()

            # zero the gradients after updating
            optimizer.zero_grad()

        Final_R2, MSE_pred, MSE_true = R_squared(test_preds, true_values_test, true_values_train)
        print("For the " + test + " We get an R squared of " + str(Final_R2) + "for the rank : " + str(rank))
        print("And for  " + test + " We get a mean squared error of  " + str(MSE_pred))
        Error_vals.append("For the " + test + " We get an R squared of " + str(Final_R2) + "for the rank : " + str(rank))



        for error in Error_vals:
            print(error)

        print("Done")