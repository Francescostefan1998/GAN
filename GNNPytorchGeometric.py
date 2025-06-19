import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_add_pool
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
dset = QM9('.')
print(len(dset))
# Here is how torch geometric wraps data
data  = dset[0]
print(data)
# It can access attributes directly
print(data.z)
# The atomic number of each atom can add attributes
data.new_attribute = torch.tensor([1,2,3])
print(data)
# can move all attributes between devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data.to(device)
data.new_attribute.is_cuda
print(data.new_attribute.is_cuda)