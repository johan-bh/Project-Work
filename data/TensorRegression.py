import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Import data here

# Use tensor regression to reduce size of 3D matrix of size 64x64x7 to single value 64 + 7 = 71

# We need to process coherence maps differently to get the right shape. We should not ravel the coherence map, but stack it.

# Too be written...