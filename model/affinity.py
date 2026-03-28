import numpy as np
import torch
from torch_geometric.utils import dropout_adj
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import GCNConv, DenseGCNConv

vector_operations = {
    "cat": (lambda x, y: torch.cat((x, y), -1), lambda dim: 2 * dim),
    "add": (torch.add, lambda dim: dim),
    "sub": (torch.sub, lambda dim: dim),
    "mul": (torch.mul, lambda dim: dim),
    "combination1": (lambda x, y: torch.cat((x, y, torch.add(x, y)), -1), lambda dim: 3 * dim),
    "combination2": (lambda x, y: torch.cat((x, y, torch.sub(x, y)), -1), lambda dim: 3 * dim),
    "combination3": (lambda x, y: torch.cat((x, y, torch.mul(x, y)), -1), lambda dim: 3 * dim),
    "combination4": (lambda x, y: torch.cat((torch.add(x, y), torch.sub(x, y)), -1), lambda dim: 2 * dim),
    "combination5": (lambda x, y: torch.cat((torch.add(x, y), torch.mul(x, y)), -1), lambda dim: 2 * dim),
    "combination6": (lambda x, y: torch.cat((torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 2 * dim),
    "combination7": (lambda x, y: torch.cat((torch.add(x, y), torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 3 * dim),
    "combination8": (lambda x, y: torch.cat((x, y, torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 4 * dim),
    "combination9": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.mul(x, y)), -1), lambda dim: 4 * dim),
    "combination10": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.sub(x, y)), -1), lambda dim: 4 * dim),
    "combination11": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 5 * dim)
}
class DenseGCNModel(torch.nn.Module):
    def __init__(self, layers_dim, edge_dropout_rate=0, supplement_mode=None):
        super(DenseGCNModel, self).__init__()

        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = DenseGCNBlock(layers_dim, 0.1, relu_layers_index=range(self.num_layers), dropout_layers_index=range(self.num_layers), supplement_mode=supplement_mode)

    def forward(self, graph, ):
        xs, adj, num_node1s, num_node2s = graph.x, graph.adj, graph.num_node1s, graph.num_node2s
        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj[indexs], p=0.1, force_undirected=True, num_nodes=num_node1s + num_node2s, training=self.training)
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout

        embeddings = self.graph_conv(xs, adj_dropout)

        return embeddings

class DenseGCNBlock(torch.nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[], supplement_mode=None):
        super(DenseGCNBlock, self).__init__()

        self.conv_layers = torch.nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            if supplement_mode is not None and i == 1:
                self.supplement_func, supplement_dim_func = vector_operations[supplement_mode]

                conv_layer_input = supplement_dim_func(gcn_layers_dim[i])
            else:
                conv_layer_input = gcn_layers_dim[i]
            conv_layer = DenseGCNConv(conv_layer_input, gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, adj, supplement_x=None):
        output = x
        embeddings = [x]
        for conv_layer_index in range(len(self.conv_layers)):
            if supplement_x is not None and conv_layer_index == 1:
                supplement_x = torch.unsqueeze(supplement_x, 0)
                output = self.supplement_func(output, supplement_x)
            output = self.conv_layers[conv_layer_index](output, adj, add_loop=False)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(torch.squeeze(output, dim=0))
        return embeddings

class ConvNet(torch.nn.Module):
    def __init__(self, ag_init_dim=2339, mg_init_dim=78, pg_init_dim=54, affinity_dropout_rate=0.2,
                 embedding_dim=128, integration_mode="combination4"):
        super(ConvNet, self).__init__()

        affinity_graph_dims = [ag_init_dim, 512, 256]


        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 2, mg_init_dim * 4]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 2, pg_init_dim * 4]

        drug_transform_dims = [affinity_graph_dims[-1], 1024, drug_graph_dims[1]]
        target_transform_dims = [affinity_graph_dims[-1], 1024, target_graph_dims[1]]

        drug_output_dims = [drug_graph_dims[-1], 1024, embedding_dim]
        target_output_dims = [target_graph_dims[-1], 1024, embedding_dim]

        self.output_dim = embedding_dim
        self.affinity_graph_conv = DenseGCNModel(affinity_graph_dims, affinity_dropout_rate)
        self.drug_transform_linear = LinearBlock(drug_transform_dims, 0.1, relu_layers_index=[0],
                                                 dropout_layers_index=[0, 1])
        self.target_transform_linear = LinearBlock(target_transform_dims, 0.1, relu_layers_index=[0],
                                                   dropout_layers_index=[0, 1])

    def forward(self, affinity_graph):

        num_node1s, num_node2s = affinity_graph.num_node1s, affinity_graph.num_node2s

        affinity_graph_embedding = self.affinity_graph_conv(affinity_graph)[-1]
        drug_transform_embedding = self.drug_transform_linear(affinity_graph_embedding[:num_node1s])[-1]
        target_transform_embedding = self.target_transform_linear(affinity_graph_embedding[num_node1s:])[-1]

        return drug_transform_embedding, target_transform_embedding


class LinearBlock(torch.nn.Module):
    def __init__(self, linear_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        embeddings = [x]
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(output)
        return embeddings