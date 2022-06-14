
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


#  ---------- COHERENCE MAP AND RESPONSE VARIABLE LOAD AND HANDLING ----------


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
    def __init__(self, rank=1):
        """
        In the constructor we instantiate 71*rank parameters and assign them as
        member parameters.
        """
        super().__init__()
        # Initialize base parameters
        self.rank = rank
        self.beta_0 = torch.nn.Parameter(torch.randn(()))
        self.CP = {}

        # Here we define our torch parameters, which is our CP variables.
        # I ended up having to do this as each variable on at a time
        # because i couldn't find a way to set a string in as the name for a variable,
        # and torch didn't accept dictionaries.

        if self.rank < 2:
            self.g00 = torch.nn.Parameter(torch.randn(()))
            self.g01 = torch.nn.Parameter(torch.randn(()))
            self.g02 = torch.nn.Parameter(torch.randn(()))
            self.g03 = torch.nn.Parameter(torch.randn(()))
            self.g04 = torch.nn.Parameter(torch.randn(()))
            self.g05 = torch.nn.Parameter(torch.randn(()))
            self.g06 = torch.nn.Parameter(torch.randn(()))

            self.a00 = torch.nn.Parameter(torch.randn(()))
            self.a01 = torch.nn.Parameter(torch.randn(()))
            self.a02 = torch.nn.Parameter(torch.randn(()))
            self.a03 = torch.nn.Parameter(torch.randn(()))
            self.a04 = torch.nn.Parameter(torch.randn(()))
            self.a05 = torch.nn.Parameter(torch.randn(()))
            self.a06 = torch.nn.Parameter(torch.randn(()))
            self.a07 = torch.nn.Parameter(torch.randn(()))
            self.a08 = torch.nn.Parameter(torch.randn(()))
            self.a09 = torch.nn.Parameter(torch.randn(()))
            self.a010 = torch.nn.Parameter(torch.randn(()))
            self.a011 = torch.nn.Parameter(torch.randn(()))
            self.a012 = torch.nn.Parameter(torch.randn(()))
            self.a013 = torch.nn.Parameter(torch.randn(()))
            self.a014 = torch.nn.Parameter(torch.randn(()))
            self.a015 = torch.nn.Parameter(torch.randn(()))
            self.a016 = torch.nn.Parameter(torch.randn(()))
            self.a017 = torch.nn.Parameter(torch.randn(()))
            self.a018 = torch.nn.Parameter(torch.randn(()))
            self.a019 = torch.nn.Parameter(torch.randn(()))
            self.a020 = torch.nn.Parameter(torch.randn(()))
            self.a021 = torch.nn.Parameter(torch.randn(()))
            self.a022 = torch.nn.Parameter(torch.randn(()))
            self.a023 = torch.nn.Parameter(torch.randn(()))
            self.a024 = torch.nn.Parameter(torch.randn(()))
            self.a025 = torch.nn.Parameter(torch.randn(()))
            self.a026 = torch.nn.Parameter(torch.randn(()))
            self.a027 = torch.nn.Parameter(torch.randn(()))
            self.a028 = torch.nn.Parameter(torch.randn(()))
            self.a029 = torch.nn.Parameter(torch.randn(()))
            self.a030 = torch.nn.Parameter(torch.randn(()))
            self.a031 = torch.nn.Parameter(torch.randn(()))
            self.a032 = torch.nn.Parameter(torch.randn(()))
            self.a033 = torch.nn.Parameter(torch.randn(()))
            self.a034 = torch.nn.Parameter(torch.randn(()))
            self.a035 = torch.nn.Parameter(torch.randn(()))
            self.a036 = torch.nn.Parameter(torch.randn(()))
            self.a037 = torch.nn.Parameter(torch.randn(()))
            self.a038 = torch.nn.Parameter(torch.randn(()))
            self.a039 = torch.nn.Parameter(torch.randn(()))
            self.a040 = torch.nn.Parameter(torch.randn(()))
            self.a041 = torch.nn.Parameter(torch.randn(()))
            self.a042 = torch.nn.Parameter(torch.randn(()))
            self.a043 = torch.nn.Parameter(torch.randn(()))
            self.a044 = torch.nn.Parameter(torch.randn(()))
            self.a045 = torch.nn.Parameter(torch.randn(()))
            self.a046 = torch.nn.Parameter(torch.randn(()))
            self.a047 = torch.nn.Parameter(torch.randn(()))
            self.a048 = torch.nn.Parameter(torch.randn(()))
            self.a049 = torch.nn.Parameter(torch.randn(()))
            self.a050 = torch.nn.Parameter(torch.randn(()))
            self.a051 = torch.nn.Parameter(torch.randn(()))
            self.a052 = torch.nn.Parameter(torch.randn(()))
            self.a053 = torch.nn.Parameter(torch.randn(()))
            self.a054 = torch.nn.Parameter(torch.randn(()))
            self.a055 = torch.nn.Parameter(torch.randn(()))
            self.a056 = torch.nn.Parameter(torch.randn(()))
            self.a057 = torch.nn.Parameter(torch.randn(()))
            self.a058 = torch.nn.Parameter(torch.randn(()))
            self.a059 = torch.nn.Parameter(torch.randn(()))
            self.a060 = torch.nn.Parameter(torch.randn(()))
            self.a061 = torch.nn.Parameter(torch.randn(()))
            self.a062 = torch.nn.Parameter(torch.randn(()))
            self.a063 = torch.nn.Parameter(torch.randn(()))

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

        if rank < 3:
            self.g10 = torch.nn.Parameter(torch.randn(()))
            self.g11 = torch.nn.Parameter(torch.randn(()))
            self.g12 = torch.nn.Parameter(torch.randn(()))
            self.g13 = torch.nn.Parameter(torch.randn(()))
            self.g14 = torch.nn.Parameter(torch.randn(()))
            self.g15 = torch.nn.Parameter(torch.randn(()))
            self.g16 = torch.nn.Parameter(torch.randn(()))

            self.a10 = torch.nn.Parameter(torch.randn(()))
            self.a11 = torch.nn.Parameter(torch.randn(()))
            self.a12 = torch.nn.Parameter(torch.randn(()))
            self.a13 = torch.nn.Parameter(torch.randn(()))
            self.a14 = torch.nn.Parameter(torch.randn(()))
            self.a15 = torch.nn.Parameter(torch.randn(()))
            self.a16 = torch.nn.Parameter(torch.randn(()))
            self.a17 = torch.nn.Parameter(torch.randn(()))
            self.a18 = torch.nn.Parameter(torch.randn(()))
            self.a19 = torch.nn.Parameter(torch.randn(()))
            self.a110 = torch.nn.Parameter(torch.randn(()))
            self.a111 = torch.nn.Parameter(torch.randn(()))
            self.a112 = torch.nn.Parameter(torch.randn(()))
            self.a113 = torch.nn.Parameter(torch.randn(()))
            self.a114 = torch.nn.Parameter(torch.randn(()))
            self.a115 = torch.nn.Parameter(torch.randn(()))
            self.a116 = torch.nn.Parameter(torch.randn(()))
            self.a117 = torch.nn.Parameter(torch.randn(()))
            self.a118 = torch.nn.Parameter(torch.randn(()))
            self.a119 = torch.nn.Parameter(torch.randn(()))
            self.a120 = torch.nn.Parameter(torch.randn(()))
            self.a121 = torch.nn.Parameter(torch.randn(()))
            self.a122 = torch.nn.Parameter(torch.randn(()))
            self.a123 = torch.nn.Parameter(torch.randn(()))
            self.a124 = torch.nn.Parameter(torch.randn(()))
            self.a125 = torch.nn.Parameter(torch.randn(()))
            self.a126 = torch.nn.Parameter(torch.randn(()))
            self.a127 = torch.nn.Parameter(torch.randn(()))
            self.a128 = torch.nn.Parameter(torch.randn(()))
            self.a129 = torch.nn.Parameter(torch.randn(()))
            self.a130 = torch.nn.Parameter(torch.randn(()))
            self.a131 = torch.nn.Parameter(torch.randn(()))
            self.a132 = torch.nn.Parameter(torch.randn(()))
            self.a133 = torch.nn.Parameter(torch.randn(()))
            self.a134 = torch.nn.Parameter(torch.randn(()))
            self.a135 = torch.nn.Parameter(torch.randn(()))
            self.a136 = torch.nn.Parameter(torch.randn(()))
            self.a137 = torch.nn.Parameter(torch.randn(()))
            self.a138 = torch.nn.Parameter(torch.randn(()))
            self.a139 = torch.nn.Parameter(torch.randn(()))
            self.a140 = torch.nn.Parameter(torch.randn(()))
            self.a141 = torch.nn.Parameter(torch.randn(()))
            self.a142 = torch.nn.Parameter(torch.randn(()))
            self.a143 = torch.nn.Parameter(torch.randn(()))
            self.a144 = torch.nn.Parameter(torch.randn(()))
            self.a145 = torch.nn.Parameter(torch.randn(()))
            self.a146 = torch.nn.Parameter(torch.randn(()))
            self.a147 = torch.nn.Parameter(torch.randn(()))
            self.a148 = torch.nn.Parameter(torch.randn(()))
            self.a149 = torch.nn.Parameter(torch.randn(()))
            self.a150 = torch.nn.Parameter(torch.randn(()))
            self.a151 = torch.nn.Parameter(torch.randn(()))
            self.a152 = torch.nn.Parameter(torch.randn(()))
            self.a153 = torch.nn.Parameter(torch.randn(()))
            self.a154 = torch.nn.Parameter(torch.randn(()))
            self.a155 = torch.nn.Parameter(torch.randn(()))
            self.a156 = torch.nn.Parameter(torch.randn(()))
            self.a157 = torch.nn.Parameter(torch.randn(()))
            self.a158 = torch.nn.Parameter(torch.randn(()))
            self.a159 = torch.nn.Parameter(torch.randn(()))
            self.a160 = torch.nn.Parameter(torch.randn(()))
            self.a161 = torch.nn.Parameter(torch.randn(()))
            self.a162 = torch.nn.Parameter(torch.randn(()))
            self.a163 = torch.nn.Parameter(torch.randn(()))

            self.CP["gamma1"] = {}

            self.CP["gamma1"]["g00"] = self.g10
            self.CP["gamma1"]["g01"] = self.g11
            self.CP["gamma1"]["g02"] = self.g12
            self.CP["gamma1"]["g03"] = self.g13
            self.CP["gamma1"]["g04"] = self.g14
            self.CP["gamma1"]["g05"] = self.g15
            self.CP["gamma1"]["g06"] = self.g16

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

        if rank < 4:
            self.g20 = torch.nn.Parameter(torch.randn(()))
            self.g21 = torch.nn.Parameter(torch.randn(()))
            self.g22 = torch.nn.Parameter(torch.randn(()))
            self.g23 = torch.nn.Parameter(torch.randn(()))
            self.g24 = torch.nn.Parameter(torch.randn(()))
            self.g25 = torch.nn.Parameter(torch.randn(()))
            self.g26 = torch.nn.Parameter(torch.randn(()))

            self.a20 = torch.nn.Parameter(torch.randn(()))
            self.a21 = torch.nn.Parameter(torch.randn(()))
            self.a22 = torch.nn.Parameter(torch.randn(()))
            self.a23 = torch.nn.Parameter(torch.randn(()))
            self.a24 = torch.nn.Parameter(torch.randn(()))
            self.a25 = torch.nn.Parameter(torch.randn(()))
            self.a26 = torch.nn.Parameter(torch.randn(()))
            self.a27 = torch.nn.Parameter(torch.randn(()))
            self.a28 = torch.nn.Parameter(torch.randn(()))
            self.a29 = torch.nn.Parameter(torch.randn(()))
            self.a210 = torch.nn.Parameter(torch.randn(()))
            self.a211 = torch.nn.Parameter(torch.randn(()))
            self.a212 = torch.nn.Parameter(torch.randn(()))
            self.a213 = torch.nn.Parameter(torch.randn(()))
            self.a214 = torch.nn.Parameter(torch.randn(()))
            self.a215 = torch.nn.Parameter(torch.randn(()))
            self.a216 = torch.nn.Parameter(torch.randn(()))
            self.a217 = torch.nn.Parameter(torch.randn(()))
            self.a218 = torch.nn.Parameter(torch.randn(()))
            self.a219 = torch.nn.Parameter(torch.randn(()))
            self.a220 = torch.nn.Parameter(torch.randn(()))
            self.a221 = torch.nn.Parameter(torch.randn(()))
            self.a222 = torch.nn.Parameter(torch.randn(()))
            self.a223 = torch.nn.Parameter(torch.randn(()))
            self.a224 = torch.nn.Parameter(torch.randn(()))
            self.a225 = torch.nn.Parameter(torch.randn(()))
            self.a226 = torch.nn.Parameter(torch.randn(()))
            self.a227 = torch.nn.Parameter(torch.randn(()))
            self.a228 = torch.nn.Parameter(torch.randn(()))
            self.a229 = torch.nn.Parameter(torch.randn(()))
            self.a230 = torch.nn.Parameter(torch.randn(()))
            self.a231 = torch.nn.Parameter(torch.randn(()))
            self.a232 = torch.nn.Parameter(torch.randn(()))
            self.a233 = torch.nn.Parameter(torch.randn(()))
            self.a234 = torch.nn.Parameter(torch.randn(()))
            self.a235 = torch.nn.Parameter(torch.randn(()))
            self.a236 = torch.nn.Parameter(torch.randn(()))
            self.a237 = torch.nn.Parameter(torch.randn(()))
            self.a238 = torch.nn.Parameter(torch.randn(()))
            self.a239 = torch.nn.Parameter(torch.randn(()))
            self.a240 = torch.nn.Parameter(torch.randn(()))
            self.a241 = torch.nn.Parameter(torch.randn(()))
            self.a242 = torch.nn.Parameter(torch.randn(()))
            self.a243 = torch.nn.Parameter(torch.randn(()))
            self.a244 = torch.nn.Parameter(torch.randn(()))
            self.a245 = torch.nn.Parameter(torch.randn(()))
            self.a246 = torch.nn.Parameter(torch.randn(()))
            self.a247 = torch.nn.Parameter(torch.randn(()))
            self.a248 = torch.nn.Parameter(torch.randn(()))
            self.a249 = torch.nn.Parameter(torch.randn(()))
            self.a250 = torch.nn.Parameter(torch.randn(()))
            self.a251 = torch.nn.Parameter(torch.randn(()))
            self.a252 = torch.nn.Parameter(torch.randn(()))
            self.a253 = torch.nn.Parameter(torch.randn(()))
            self.a254 = torch.nn.Parameter(torch.randn(()))
            self.a255 = torch.nn.Parameter(torch.randn(()))
            self.a256 = torch.nn.Parameter(torch.randn(()))
            self.a257 = torch.nn.Parameter(torch.randn(()))
            self.a258 = torch.nn.Parameter(torch.randn(()))
            self.a259 = torch.nn.Parameter(torch.randn(()))
            self.a260 = torch.nn.Parameter(torch.randn(()))
            self.a261 = torch.nn.Parameter(torch.randn(()))
            self.a262 = torch.nn.Parameter(torch.randn(()))
            self.a263 = torch.nn.Parameter(torch.randn(()))

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
                for i in range(64):
                    if i > j:
                        CPd_sum = 0

                        for d in range(self.rank):

                            CPd_sum += self.CP["alpha" + str(d)]["a"+str(d)+str(i)] \
                                       * self.CP["alpha" + str(d)]["a"+str(d)+str(i)] \
                                       * self.CP["gamma" + str(d)]["g"+str(d)+str(k)]

                        finalsum += x[k][i][j] * CPd_sum

        return self.beta_0 + finalsum
# ----- Getting R-squared -----


def R_squared(y_pred_vals, y_true_vals):

    mean_pred = sum(y_pred_vals)/len(y_pred_vals)
    mean_true = sum(y_true_vals) / len(y_true_vals)

    MSE_pred = 0
    MSE_true = 0

    for i in y_pred_vals:
        MSE_pred += (i-mean_true)**2
    for i in y_pred_vals:
        MSE_true += (i-mean_true)**2

    R2 = 1-(MSE_pred)/(MSE_true)

    return R2

# ----- Getting Mean squared error -----
def MSEfunc(y_pred_vals, y_true_vals):

    differences = y_pred_vals-y_true_vals

    deviation = sum(differences**2)/len(y_pred_val)


    return deviation

# ---------- MODEL TRAINING ----------

tests = list(response_variables.keys())

#train_keys, test_keys = train_test_split(keys, test_size=0.33, random_state=42)

random.seed(420)

random.shuffle(keys)

for test_type in tests:
    model = Candemann_Parafac_module(rank=1)

    # 2) Define loss and optimizer
    learning_rate = 0.01
    n_iters = len(data)
    train_stop = 250

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    train_preds = []
    true_values_train = []
    test_preds = []
    true_test = []

    for epoch in range(n_iters):
        print(test_type + " : " + str(epoch))
        # predict = forward pass with our model
        y_predicted = model(data[keys[epoch]])


        # loss
        try:
            l = loss(torch.tensor(response_variables[test_type][keys[epoch]], dtype=torch.float64), y_predicted)
        except:
            continue
        #summ += response_variables[test_type][keys[epoch]]

        #Values for R squared
        if epoch < train_stop:
            train_preds.append(y_predicted)
            true_values_train.append(response_variables[test_type][keys[epoch]])
        else:
            test_preds.append(y_predicted)
            true_test.append(response_variables[test_type][keys[epoch]])

        # calculate gradients = backward pass
        l.backward()

        # update weights
        optimizer.step()

        # zero the gradients after updating
        optimizer.zero_grad()

    Final_R2 = R_squared(test_preds,true_values_train)
    MeanSquarErr = MSEfunc(test_preds,true_test)
    print("For the " + test_type + " We get an R squared of " + str(Final_R2))
    print("And for  " + test_type + " We get a mean squared error of  " + str(MeanSquarErr))
print("Done")

#  ---------- CURRENTLY UNUSED CODE  ----------


"""

# Linear regression
# f = w * x

# here : f = 2 * x

# 0) Training samples, watch the shape!
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'#samples: {n_samples}, #features: {n_features}')
# 0) create a test sample
X_test = torch.tensor([5], dtype=torch.float32)

# Here we design the model, this has to be done in order
# to specify our own cost function and cost function parameters.

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
"""

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

