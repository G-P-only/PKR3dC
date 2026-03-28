import json
from collections import OrderedDict
import os
import networkx as nx
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDConfig
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset


max_seq_len = 1000
max_smi_len = 100

thre = 16640


CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21,
               "V": 22, "Y": 23, "X": 24,
               "Z": 25, "J":26}

CHARPROTLEN = 26

def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):  # x, smi_ch_ind, y
        X[i] = smi_ch_ind[ch]

    return X


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

def atom_features(atom):
    encoding = one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
    encoding += one_of_k_encoding(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10]) + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4,5,6,7,8,9,10])
    encoding += one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6,7,8,9,10])
    encoding += one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'other'])
    encoding += [atom.GetIsAromatic()]

    try:
        encoding += one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except:
        encoding += [0, 0] + [atom.HasProp('_ChiralityPossible')]

    return np.array(encoding)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


class GNNDataset(InMemoryDataset):
    def __init__(self, root, index=0, types='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['kiba_cold_pairs_fold_1.csv', 'kiba_cold_pairs_fold_2.csv', 'kiba_cold_pairs_fold_3.csv',
                'kiba_cold_pairs_fold_4.csv', 'kiba_cold_pairs_fold_5.csv']
    @property
    def processed_file_names(self):
        return ['processed_data_fold_0.pt', 'processed_data_fold_1.pt', 'processed_data_fold_2.pt', 'processed_data_fold_3.pt','processed_data_fold_4.pt',
        'processed_test_data_fold_0.pt', 'processed_test_data_fold_1.pt', 'processed_test_data_fold_2.pt', 'processed_test_data_fold_3.pt','processed_test_data_fold_4.pt']

    def download(self):
        pass


    def process(self):
        fold_0_list = self.process_train_data(self.raw_paths[0])
        fold_1_list = self.process_train_data(self.raw_paths[1])
        fold_2_list = self.process_train_data(self.raw_paths[2])
        fold_3_list = self.process_train_data(self.raw_paths[3])
        fold_4_list = self.process_train_data(self.raw_paths[4])

        fold_0_test_list = self.process_data(self.raw_paths[0])
        fold_1_test_list = self.process_data(self.raw_paths[1])
        fold_2_test_list = self.process_data(self.raw_paths[2])
        fold_3_test_list = self.process_data(self.raw_paths[3])
        fold_4_test_list = self.process_data(self.raw_paths[4])

        print('Graph construction done. Saving to file.')

        data, slices = self.collate(fold_0_list)
        torch.save((data, slices), self.processed_paths[0])
        data, slices = self.collate(fold_1_list)
        torch.save((data, slices), self.processed_paths[1])
        data, slices = self.collate(fold_2_list)
        torch.save((data, slices), self.processed_paths[2])
        data, slices = self.collate(fold_3_list)
        torch.save((data, slices), self.processed_paths[3])
        data, slices = self.collate(fold_4_list)
        torch.save((data, slices), self.processed_paths[4])

        # save preprocessed test data:
        data, slices = self.collate(fold_0_test_list)
        torch.save((data, slices), self.processed_paths[5])
        data, slices = self.collate(fold_1_test_list)
        torch.save((data, slices), self.processed_paths[6])
        data, slices = self.collate(fold_2_test_list)
        torch.save((data, slices), self.processed_paths[7])
        data, slices = self.collate(fold_3_test_list)
        torch.save((data, slices), self.processed_paths[8])
        data, slices = self.collate(fold_4_test_list)
        torch.save((data, slices), self.processed_paths[9])

    def process_data(self, data_path):
        df = pd.read_csv(data_path)
        data_list = []

        proteins = json.load(
            open(os.path.join('D:\PythonProject\PKR3dC\data\kiba', 'proteins.txt')),
            object_pairs_hook=OrderedDict)
        prots = []  # sequences
        prot_keys = []
        for t in proteins.keys():
            prots.append(proteins[t])
            prot_keys.append(t)

        for i, row in df.iterrows():
            smi = row['SMILES']
            sequence = row['target_sequence']
            affinity = row['affinity']
            did = row['drug_id']
            pid = row['protein_id']
            mol = Chem.MolFromSmiles(smi)
            c_size, features, edge_index = smile_to_graph(smi)
            for pair_ind in range(len(prots)):
                if sequence == prots[pair_ind]:
                    a = (prot_keys[pair_ind])
                    break
            target_key = a
            target = seq_cat(sequence)

            data = DATA.Data(
                x=torch.FloatTensor(features),
                # smi=smi,
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                # edge_attr=edge_attr,
                y=torch.FloatTensor([affinity]),
                target=torch.LongTensor([target]),
                c_size=torch.LongTensor([c_size]),
                protein_3d=target_key,
                did=torch.FloatTensor([did]),
                pid=torch.LongTensor([pid]),

            )
            data_list.append(data)

        return data_list

    def process_train_data(self, data_path):
        df = pd.read_csv(data_path)
        data_list = []

        proteins = json.load(

            open(os.path.join('D:\PythonProject\PKR3dC\data\kiba', 'proteins.txt')),
            object_pairs_hook=OrderedDict)

        prots = [] #229 # sequences
        prot_keys = []

        for t in proteins.keys():
            prots.append(proteins[t])
            prot_keys.append(t)
        for i, row in df.iterrows():
            smi = row['SMILES']
            sequence = row['target_sequence']
            affinity = row['affinity']
            did = row['drug_id']
            pid = row['protein_id']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue

            c_size, features, edge_index = smile_to_graph(smi)
            target = seq_cat(sequence)
            for pair_ind in range(len(prots)):#229
                if sequence == prots[pair_ind]:
                    a = (prot_keys[pair_ind])
                    break
            target_key = a

            data = DATA.Data(
                x=torch.FloatTensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.FloatTensor([affinity]),
                target=torch.LongTensor([target]),
                c_size = torch.LongTensor([c_size]),
                protein_3d=target_key,
                did = torch.FloatTensor([did]),
                pid = torch.LongTensor([pid]),

            )
            data_list.append(data)

        return data_list

if __name__ == "__main__":
    dataset = 'kiba'
    GNNDataset('D:\PythonProject\PKR3dC\data\{}\\feicold'.format(dataset))
    GNNDataset('D:\PythonProject\PKR3dC\data\{}\drug'.format(dataset))
    GNNDataset('D:\PythonProject\PKR3dC\data\{}\protein'.format(dataset))
    GNNDataset('D:\PythonProject\PKR3dC\data\{}\pair'.format(dataset))


