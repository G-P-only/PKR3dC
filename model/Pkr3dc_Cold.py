import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp
from model.MolTrans import MTs_config, Moltrans

from model.ProVAE import CNN, decoder
from model.PairMap import BiAttention
from model.ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
from model.gvp.models import ThreeD_Protein_Model, TransformerGVP
from .DenseNet import GraphDenseNet
from model.affinity import DenseGCNModel, LinearBlock

MTconfig = MTs_config()


class DTF(nn.Module):
    def __init__(self, channels=206, r=4):
        super(DTF, self).__init__()
        inter_channels = int(channels // r)  # 51

        self.att1 = nn.Sequential(
            nn.Linear(channels, inter_channels),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels),
            nn.BatchNorm1d(channels)
        )

        self.att2 = nn.Sequential(
            nn.Linear(channels, inter_channels),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels),
            nn.BatchNorm1d(channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, fd, fp):
        w1 = self.sigmoid(self.att1(fd + fp))
        fout1 = fd * w1 + fp * (1 - w1)

        w2 = self.sigmoid(self.att2(fout1))
        fout2 = fd * w2 + fp * (1 - w2)
        return fout2


class PKR3dC(torch.nn.Module):
    def __init__(self, num_features_xd=87, n_output=1, num_features_xt=25, n_filters=32, embed_dim=128, output_dim=128,
                 dropout=0.2, affinity_graph=None, ag_init_dim=2342, dataset=None, cold=None):
        super(PKR3dC, self).__init__()
        self.device = torch.device('cuda:0')
        self.dataset = dataset
        self.cold = cold
        self.affinity_graph = affinity_graph.to(self.device)
        self.ligand_encoder = GraphDenseNet(num_input_features=87, out_dim=32 * 4, block_config=[8, 8, 8],
                                            bn_sizes=[2, 2, 2])
        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.ProCNN = CNN(32, 7).to(self.device)
        self.ProCNN.to(self.device)
        self.protein_model = ThreeD_Protein_Model(node_in_dim=(6, 3), node_h_dim=(128, 32), edge_in_dim=(32, 1),
                                                  edge_h_dim=(32, 1),
                                                  seq_in=True, num_layers=3, drop_rate=0.1, attention_type="performer")
        self.fc_xt1 = Linear(32 * 121, output_dim)


        self.dtf = DTF()
        # combined layers
        self.fc1 = Linear(384, 1024)
        self.fc2 = Linear(1024, 256)
        self.out = Linear(256, n_output)

        # self.gcs_attention = BiAttention(3*embed_dim,embed_dim,4).to(self.device)
        # affinity
        if self.dataset == "davis":
            self.filename = "Davis"
        else:
            self.filename = "KIBA"
        affinity_graph_dims = [ag_init_dim, 512, 256]
        affinity_dropout_rate = 0.2
        drug_transform_dims = [affinity_graph_dims[-1], 1024, 78]
        target_transform_dims = [affinity_graph_dims[-1], 1024, 216]
        self.affinity_graph_conv = DenseGCNModel(affinity_graph_dims, affinity_dropout_rate)
        self.drug_transform_linear = LinearBlock(drug_transform_dims, 0.1, relu_layers_index=[0],
                                                 dropout_layers_index=[0, 1])
        self.target_transform_linear = LinearBlock(target_transform_dims, 0.1, relu_layers_index=[0],
                                                   dropout_layers_index=[0, 1])

        self.dtf = DTF()
        self.linear_protein = torch.nn.Linear(344, 206)

        self.classifier = nn.Sequential(
            nn.Linear(206, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # activation and regularization
        self.relu = ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, drug_map=None, drug_weight=None, protein_map=None, protein_weight=None):
        # graph input
        x = self.ligand_encoder(data)  # [batch,128]

        # protein input feed-forward:
        target = data.target
        embedded_xt = self.embedding_xt(target)
        embedded_xt = embedded_xt.permute(0, 2, 1)
        out_t, mu_t, logvar_t = self.ProCNN(embedded_xt)
        out_t = out_t.permute(0, 2, 1)

        name3d_list = data.protein_3d
        protein_graphs = {}
        for idx, name3d in enumerate(name3d_list):
            protein_file = os.path.join(
                f"D:\PythonProject\secondMGR\data\{self.dataset}\prot_3d_for_{self.filename}\\res_graph\\{name3d}.pdb.pt")
            protein_graph = torch.load(protein_file, map_location="cpu")

            protein_graphs[idx] = protein_graph

        data_list = list(protein_graphs.values())
        protein_batch = Batch.from_data_list(data_list)
        node_counts = [data.num_nodes for data in data_list]
        batch = [idx for idx, count in enumerate(node_counts) for _ in range(count)]
        protein_batch.batch = torch.tensor(batch)
        device = torch.device('cuda:0')
        protein_batch = protein_batch.to(device)
        x_prot = self.protein_model((protein_batch.node_s, protein_batch.node_v),
                                    protein_batch.edge_index, (protein_batch.edge_s, protein_batch.edge_v),
                                    protein_batch.seq, protein_batch.batch)
        protein = torch.cat([out_t, x_prot], dim=1)
        protein = protein.mean(dim=1)

        # affinity_graph
        num_node1s, num_node2s = self.affinity_graph.num_node1s, self.affinity_graph.num_node2s
        affinity_graph_embedding = self.affinity_graph_conv(self.affinity_graph)[-1]
        if drug_map is not None:
            if drug_weight is not None:
                drug_weight = torch.FloatTensor(drug_weight).to(device)
                drug_affinity_graph = torch.sum(self.drug_transform_linear(affinity_graph_embedding[:num_node1s])[-1][drug_map, :] * drug_weight, dim=-2)
            else:
                drug_affinity_graph = torch.mean(self.drug_transform_linear(affinity_graph_embedding[:num_node1s])[-1][drug_map, :], dim=-2)
        else:
            drug_affinity_graph = self.drug_transform_linear(affinity_graph_embedding[:num_node1s])[-1]
        if protein_map is not None:
            if protein_weight is not None:
                protein_weight = torch.FloatTensor(protein_weight).to(device)
                protein_affinity_graph = torch.sum(self.target_transform_linear(affinity_graph_embedding[num_node1s:])[-1][protein_map, :] * protein_weight, dim=-2)
            else:
                protein_affinity_graph = torch.mean(self.target_transform_linear(affinity_graph_embedding[num_node1s:])[-1][protein_map, :], dim=-2)
        else:
            protein_affinity_graph = self.target_transform_linear(affinity_graph_embedding[num_node1s:])[-1]

        did = data.did.long()
        pid = data.pid.long()
        did, pid = self.did_index_dict(did, pid, drug_map, protein_map)
        draff = drug_affinity_graph[did]
        proaff = protein_affinity_graph[pid]

        findr = torch.cat((x, draff), dim=1)
        finpro = torch.cat((protein, proaff), dim=1)
        finpro = self.linear_protein(finpro)
        x = self.dtf(findr, finpro)

        x = self.classifier(x)
        return x

    def did_index_dict(self, did, pid, drug_map, protein_map):
        if (drug_map is not None) or (protein_map is not None):
            if self.cold == 'drug':
                did = [self.affinity_graph.test_did_index_dict.get(val.item(), -1) for val in did]
            elif self.cold == 'protein':
                pid = [self.affinity_graph.test_pid_index_dict.get(val.item(), -1) for val in pid]
            else:
                did = [self.affinity_graph.drug_test_did_index_dict.get(val.item(), -1) for val in did]
                pid = [self.affinity_graph.protein_test_pid_index_dict.get(val.item(), -1) for val in pid]
        elif self.cold != 'feicold':
            if self.cold == 'drug':
                did = [self.affinity_graph.train_did_index_dict.get(val.item(), -1) for val in did]
            elif self.cold == 'protein':
                pid = [self.affinity_graph.train_pid_index_dict.get(val.item(), -1) for val in pid]
            else:
                did = [self.affinity_graph.drug_train_did_index_dict.get(val.item(), -1) for val in did]
                pid = [self.affinity_graph.protein_train_pid_index_dict.get(val.item(), -1) for val in pid]
        return did, pid