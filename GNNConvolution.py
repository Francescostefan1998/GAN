#pip install networkx
import numpy as np 
import networkx as nx
G = nx.Graph()
# Hex codes for colors if we draw graph
blue, orange, green = "#1f77b4", "#ff7f0e", "#2ca02c"
G.add_nodes_from([
    (1, {"color": blue}),
    (2, {"color": orange}),
    (3, {"color": blue}),
    (4, {"color": green})
])

G.add_edges_from([(1,2), (2,3), (1,3), (3,4)])
A = np.asarray(nx.adjacency_matrix(G).todense())
print(A)

def build_graph_color_label_representation(G, mapping_dict):
    one_hot_idxs = np.array([mapping_dict[v] for v in
                             nx.get_node_attributes(G, 'color').values()])
    one_hot_encoding = np.zeros((one_hot_idxs.size, len(mapping_dict)))
    one_hot_encoding[np.arange(one_hot_idxs.size), one_hot_idxs] = 1
    return one_hot_encoding

X = build_graph_color_label_representation(G, {green:0, blue:1, orange:2})
print(X)

color_map = nx.get_node_attributes(G, 'color').values()
nx.draw(G, with_labels = True, node_color=color_map)

f_in, f_out = X.shape[1], 6
W_1 = np.random.rand(f_in, f_out)
W_2 = np.random.rand(f_in, f_out)
h = np.dot(X, W_1) + np.dot(np.dot(A, X), W_2)

# Defining the NodeNetwork model
import networkx as nx
import torch
from torch.nn.parameter import Parameter
import numpy as np
import math
import torch.nn.functional as F

class NodeNetwork(torch.nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.conv_1 = BasicGraphConvolutionLayer(input_features, 32) # here is where the magic formula should happen
        self.conv_2 = BasicGraphConvolutionLayer(32, 32)
        self.fc_1 = torch.nn.Linear(32, 16)
        self.out_layer = torch.nn.Linear(16, 2)

    def forward(self, X, A, batch_mat):
        x = F.relu(self.conv_1(X, A))
        x = F.relu(self.conv_2(x, A))
        output = global_sum_pool(x, batch_mat)
        output = self.fc_1(output)
        output = self.out_layer(output)
        return F.softmax(output, dim=1)
    

class BasicGraphConvolutionLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W2 = Parameter(torch.rand((in_channels, out_channels), dtype=torch.float32))
        self.W1 = Parameter(torch.rand((in_channels, out_channels), dtype=torch.float32))
        self.bias = Parameter(torch.zeros(out_channels, dtype = torch.float32))
    def forward(self, X, A):
        potential_msgs = torch.mm(X, self.W2)
        propagated_msgs = torch.mm(A, potential_msgs)
        root_update = torch.mm(X, self.W1)
        output = propagated_msgs + root_update + self.bias
        return output

print('X.shape:', X.shape)
print('A.shape:', A.shape)
basiclayer = BasicGraphConvolutionLayer(3, 8)
out = basiclayer(X = torch.tensor(X, dtype=torch.float32), A = torch.tensor(A, dtype=torch.float32))
print('Output shape:', out.shape)

def global_sum_pool(X, batch_mat):
    if batch_mat is None or batch_mat.dime() == 1:
        return torch.sum(X, dim=0).unsqueeze(0)
    else:
        return torch.mm(batch_mat, X)

def get_batch_tensor(graph_sizes): # graph_sizes a list with the number of node in each graph in a batch
    starts = [sum(graph_sizes[:idx]) for idx in range(len(graph_sizes))] # calculate the starting index of the graph based from the concatenated node 
    stops = [starts[idx] + graph_sizes[idx] for idx in range(len(graph_sizes))] # same for the stopping index
    tot_len = sum(graph_sizes) # total number of nodes in the whole batch
    batch_size = len(graph_sizes) # number of graph in the batch
    batch_mat = torch.zeros([batch_size, tot_len]).float() # creating a 0 matrix of shape [batch_isze, total nodes]
    for idx, starts_and_stops in enumerate(zip(starts, stops)):
        start = starts_and_stops[0]
        stop = starts_and_stops[1]
        batch_mat[idx, start:stop] = 1 # set the corresponding node for that graph to one and the other to 0
    return batch_mat

# batch is  a list of dictionaries each containing the representation and label of a graph

def collate_graphs(batch):
    adj_mats = [graph['A'] for graph in batch] # Get all the adjacency matrix and put them in a list
    sizes = [A.size(0) for A in adj_mats] # Get the number of nodes in each graph
    tot_size = sum(sizes) # Total number of node across all the batches in the graph
    # create batch matrix
    batch_mat = get_batch_tensor(sizes) # build the batch matrix (function above)
    # combine feature matrix
    feat_mats = torch.cat([graph['X'] for graph in batch], dim = 0) # Concatenate all the feature matrices X from each graph into [tot_size, feature_dimension] matrix
    # combine labels
    labels = torch.zeros([tot_size, tot_size], dtype=torch.float32) # Create a placeholder labels
    # combine adjacency matrices
    batch_adj = torch.zeros([tot_size, tot_size], dtype=torch.float32) # Initialize a big matrix to hold all the Adjacency matrix in the correct block position
    accum = 0
    for adj in adj_mats:
        g_size = adj.shape[0] # gets its size
        batch_adj[accum:accum+g_size, accum:accum+g_size] = adj # insert it into the right block in the big batch_adj matrix
        accum = accum + g_size # update it to shift to the next block
    repr_and_label = {'A': batch_adj, 'X': feat_mats, 'y': labels, 'batch': batch_mat} # pack everything into a dictionary
    return repr_and_label # return the batch ready data

# builds a dictionary representation that we will use later
def get_graph_dict(G, mapping_dict):
    # Function builds dictionary representation of graph G
    A = torch.from_numpy(np.asarray(nx.adjacency_matrix(G).todense())).float()
    # build_graph_color_label_representation()
    # was introduced with the first example graph
    X = torch.from_numpy(build_graph_color_label_representation(G, mapping_dict)).float()
    # kludge since there is not specific task for this example
    y = torch.tensor([[1, 0]]).float()
    return {'A':A, 'X':X, 'y':y, 'batch':None}

# building 4 graphs to threat as a dataset
blue, orange, green = "#1f77b4", "#ff7f0e",  "#2ca02c"
mapping_dict = {green:0, blue:1, orange:2}
G1 = nx.Graph()
G1.add_nodes_from([
    (1, {"color": blue}),
    (2, {"color": orange}),
    (3, {"color": blue}),
    (4, {"color": green})
])
G1.add_edges_from([(1,2), (2,3), (1,3), (3,4)])
G2 = nx.Graph()
G2.add_nodes_from([
    (1, {"color": green}),
    (2, {"color": green}),
    (3, {"color": orange}),
    (4, {"color": orange}),
    (5, {"color": blue})
])
G2.add_edges_from([(2,3), (3,4), (3,1), (5,1)])
G3 = nx.Graph()
G3.add_nodes_from([
    (1, {"color": orange}),
    (2, {"color": orange}),
    (3, {"color": green}),
    (4, {"color": green}),
    (5, {"color": blue}),
    (6, {"color": orange})
])
G3.add_edges_from([(2,3), (3,4), (3,1), (5,1), (2,5), (6,1)])
G4 = nx.Graph()
G4.add_nodes_from([
    (1, {"color": blue}),
    (2, {"color": blue}),
    (3, {"color": green})
])
G4.add_edges_from([(1,2), (2,3)])
graph_list = [get_graph_dict(graph, mapping_dict) for graph in [G1, G2, G3, G4]]