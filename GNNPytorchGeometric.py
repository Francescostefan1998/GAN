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

class ExampleNet(torch.nn.Module):  
    def __init__(self, num_node_features, num_edge_features):
        super().__init__()

        # MLP for conv1: maps edge features to a (num_node_features x 32) weight matrix
        # Used to generate filters dynamically per edge in the first NNConv
        conv1_net = nn.Sequential(
            nn.Linear(num_edge_features, 32),  # edge feature input → hidden size
            nn.ReLU(),                         # non-linearity
            nn.Linear(32, num_node_features * 32)  # output a flattened weight matrix
        )

        # MLP for conv2: maps edge features to a (32 x 16) weight matrix
        # Same idea as above but for the second layer
        conv2_net = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32 * 16)
        )

        # First NNConv layer:
        # Input: node features of dim `num_node_features`
        # Output: 32-dimensional node embeddings
        self.conv1 = NNConv(num_node_features, 32, conv1_net)

        # Second NNConv layer:
        # Input: 32-dim node features from previous layer
        # Output: 16-dimensional node embeddings
        self.conv2 = NNConv(32, 16, conv2_net)

        # Fully connected layer to process graph-level representation
        # Usually applied after pooling (like global_add_pool)
        self.fc_1 = nn.Linear(16, 32)

        # Final output layer (e.g., for regression or binary classification)
        self.out = nn.Linear(32, 1)

    def forward(self, data):
        batch, x, edge_index, edge_attr = (
            data.batch, data.x, data.edge_index, data.edge_attr
        )
        # first graph conv layer
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        # second graph conv layer
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc_1(x))
        output = self.out(x)
        return output
    

from torch.utils.data import random_split
train_set, valid_set, test_set = random_split(dset, [110000, 10831, 10000])
trainloader = DataLoader(train_set, batch_size=32, shuffle=True)
validloader = DataLoader(valid_set, batch_size=32, shuffle=True)
testloader = DataLoader(test_set, batch_size=32, shuffle=True)

# initialize network
qm9_node_feats, qm9_edge_feats = 11, 4
net = ExampleNet(qm9_node_feats, qm9_edge_feats)
# initialize an optimizer with some reasonable parameters
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
epochs = 4
target_idx = 1 # index position of the polarizability label
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

for total_epochs in range(epochs):
    epoch_loss = 0
    total_graphs = 0
    net.train()
    for batch in trainloader:
        batch.to(device)
        optimizer.zero_grad()
        output = net(batch)
        loss = F.mse_loss(output, batch.y[:, target_idx].unsqueeze(1))
        loss.backward()
        epoch_loss += loss.item()
        total_graphs += batch.num_graphs
        optimizer.step()
    train_avg_loss = epoch_loss / total_graphs
    val_loss = 0
    total_graphs = 0
    net.eval()
    for batch in validloader:
        batch.to(device)
        output = net(batch)
        loss = F.mse_loss(output, batch.y[:, target_idx].unsqueeze(1))
        val_loss += loss.item()
        total_graphs += batch.num_graphs
    val_avg_loss = val_loss / total_graphs
    print(f"Epochs: {total_epochs} | epoch avg. loss: {train_avg_loss:.2f} | validation avg. loss: {val_avg_loss:.2f}")

net.eval()
predictions = []
real = []
for batch in testloader:
    output = net(batch.to(device))
    predictions.append(output.detach().cpu().numpy())
    real.append(batch.y[:, target_idx].detach().cpu().numpy())

real = np.concatenate(real)
predictions = np.concatenate(predictions)

# plot the first 500 predictions
import matplotlib.pyplot as plt
plt.scatter(real[:500], predictions[:, 500])
plt.xlabel('Isotropic polarizability')
plt.ylabel('Predicted isotropic polarizability')