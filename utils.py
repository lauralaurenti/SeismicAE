import torch
import random
import os
import numpy as np
from geopy.distance import geodesic
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

import networkx as nx
from networkx.classes.function import degree
from networkx.linalg.laplacianmatrix import normalized_laplacian_matrix

def seed_everything(seed):
    """It sets all the seeds for reproducibility.

    Args:
    ----------
    seed : int
        Seed for all the methods
    """
    print("Setting seeds")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def graph_creator(station_choice, cutoff):
    print(station_choice, cutoff)
    if station_choice == 'network_1':
        stations = pd.read_pickle('data/chosenStations.pkl')
        stations = stations[['Station','Latitude','Longitude']]
        print(stations.shape)
        print(stations.head(2))
        
    else:
        stations = pd.read_csv('data/othernetwork/stationDists.csv')
        stations = stations[['sta','lat', 'lon']]
        stations.columns = ['Station','Latitude','Longitude']
        print(stations.shape)
        print(stations.head(2))
    
    station_coords = stations[['Latitude','Longitude']].values
    
    graph = nx.Graph()

    for k in stations[['Station','Longitude','Latitude']].iterrows():
        graph.add_node(k[1].iloc[0], pos=(k[1].iloc[1],k[1].iloc[2]))

    distances = []

    for idx1, itm1 in stations[['Station','Longitude','Latitude']].iterrows():
            for idx2, itm2 in stations[['Station','Longitude','Latitude']].iterrows():
                    pos1 = (itm1.iloc[1],itm1.iloc[2])
                    pos2 = (itm2.iloc[1],itm2.iloc[2])
                    distance = geodesic(pos1, pos2,).km #geopy distance
                    if distance != 0: # this filters out self-loops and also the edges between the artificial nodes
                        graph.add_edge(itm1.iloc[0], itm2.iloc[0], weight=distance, added_info=distance)

    names = []
    for i in graph.nodes():
        names.append(i)
    indexes = [i for i in range(0,39)]
    zip_iterator = zip(indexes, names)
    a_dictionary = dict(zip_iterator)

    edge_list = nx.to_pandas_edgelist(graph)
    edge_list['weight'] = (edge_list['weight'] - min(edge_list['weight'])) / (max(edge_list['weight']) - min(edge_list['weight']))
    edge_list['weight'] = 0.98 - edge_list['weight']
    adj = nx.from_pandas_edgelist(edge_list, edge_attr=['weight'])#,source=['source'], target=['target'])
    adj = pd.DataFrame(nx.adjacency_matrix(adj, weight='weight').todense())
    adj[adj < cutoff] = 0
    return torch.tensor(adj.values)

def normalize_for_emb(inputs):
    print("The normalization is event-based, working on the 3 channels")
    # in_max = np.max(np.abs(inputs.reshape(-1, inputs.shape[2])), axis=1, keepdims=True)
    in_max = np.max(np.abs(inputs.reshape(inputs.shape[0]*inputs.shape[1], inputs.shape[2]*inputs.shape[3])), axis=1, keepdims=True)
    in_max[in_max == 0.0] = 1e-10
    # in_norm = inputs.reshape(-1, inputs.shape[2]) / in_max
    in_norm = inputs.reshape(inputs.shape[0]*inputs.shape[1], inputs.shape[2]*inputs.shape[3]) / in_max
    inputs_norm = np.reshape(in_norm, inputs.shape).astype(np.float32)
    return inputs_norm

def normalize(inputs):
    print("The normalization works on all the 39 stations and the 3 channels")
    normalized = []
    for eq in inputs:
        maks = np.max(np.abs(eq))
        if maks != 0:
            normalized.append(eq/maks)
        else:
            normalized.append(eq)
    return np.array(normalized)




class OriginalModel_cnn(nn.Module):
    def __init__(self):
        super(OriginalModel_cnn, self).__init__()
        self.model_chosen = 'nofeatures'
        reg_const = 0.0001
        self.relu = nn.ReLU()
        self.trace_len =1000
        self.conv1 = nn.Conv2d(3, 32, (1, 125), stride=(1, 2), padding=(0, 0))
        self.conv2 = nn.Conv2d(32, 64, (1, 125), stride=(1, 2), padding=(0, 0))
        self.conv3 = nn.Conv2d(64, 64, (39, 5), stride=(39, 5), padding=(19, 2))
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        self.dense1 = nn.Linear(64 * 2 * 39, 128)

        self.graph_features_flattened = nn.Flatten()
        if self.trace_len==1000:
            dim_here=2048
        elif self.trace_len==2500:
            dim_here=6848
        else:
            print("trace_len not valid")

        if self.model_chosen == 'nofeatures':
            self.merged_dense = nn.Linear(dim_here, 128)
        if self.model_chosen == 'main':
            self.merged_dense = nn.Linear(dim_here + 78, 128)

        self.pga = nn.Linear(128, 39)
        self.pgv = nn.Linear(128, 39)
        self.sa03 = nn.Linear(128, 39)
        self.sa10 = nn.Linear(128, 39)
        self.sa30 = nn.Linear(128, 39)

    def forward(self, wav_input, graph_features, graph_input):
        model_chosen = self.model_chosen
        wav_input=wav_input.permute(0,3,1,2)
        # print("in" ,wav_input.shape)
        conv1_output = self.relu(self.conv1(wav_input))
        # print("conv1" ,conv1_output.shape)
        conv2_output = self.relu(self.conv2(conv1_output))
        # print("conv2" ,conv2_output.shape)
        conv3_output = self.relu(self.conv3(conv2_output))
        # print("conv3" ,conv3_output.shape)
        

        conv3_output_flattened = self.flatten(conv3_output)
        # print("conv3 flat" ,conv3_output_flattened.shape)
        
        conv3_output_flattened_dropout = self.dropout(conv3_output_flattened)
        # print("conv3 flat drop" ,conv3_output_flattened_dropout.shape)
        

        graph_features_flattened = self.graph_features_flattened(graph_features)
        # print("graph_features_flattened" ,graph_features_flattened.shape)
        

        if model_chosen == 'nofeatures':
            merged = self.relu(self.merged_dense(conv3_output_flattened_dropout))
        if model_chosen == 'main':
            merged = torch.cat((conv3_output_flattened_dropout, graph_features_flattened), dim=1)
            merged = self.relu(self.merged_dense(merged))
        # print("merged" ,merged.shape)
        
        pga_output = self.pga(merged)
        pgv_output = self.pgv(merged)
        sa03_output = self.sa03(merged)
        sa10_output = self.sa10(merged)
        sa30_output = self.sa30(merged)

        return pga_output, pgv_output, sa03_output, sa10_output, sa30_output



class OriginalModel_gcn(nn.Module):
    def __init__(self):
        super(OriginalModel_gcn, self).__init__()
        self.model_chosen = 'nofeatures'
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 125), stride=(1, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 125), stride=(1, 2))
        
        self.reshape = None  # To be determined dynamically
        gcn_input_dim = 64 * ((39 // 4) // 39)  # Approximate reshaped width after conv layers

        self.gcn1 = GCNConv(gcn_input_dim + 2 if self.model_chosen == 'main' else gcn_input_dim, 64)
        self.gcn2 = GCNConv(64, 64)
        
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(2496, 128)
        self.pga = nn.Linear(128, 39)
        self.pgv = nn.Linear(128, 39)
        self.sa03 = nn.Linear(128, 39)
        self.sa10 = nn.Linear(128, 39)
        self.sa30 = nn.Linear(128, 39)
        
        
    def forward(self, wav_input, graph_features, graph_input):
        wav_input=wav_input.permute(0,3,1,2)
        x = F.relu(self.conv1(wav_input))
        x = F.relu(self.conv2(x))
        batch_size= x.shape[0]
        x = x.reshape(batch_size,39,-1)
        edge_index, edge_weight = dense_to_sparse(torch.Tensor(graph_input[0]))
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x = F.relu(self.gcn1(data.x,  data.edge_index, data.edge_attr))
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x = torch.tanh(self.gcn2(data.x,  data.edge_index, data.edge_attr))
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))

        pga = self.pga(x)
        pgv = self.pgv(x)
        sa03 = self.sa03(x)
        sa10 = self.sa10(x)
        sa30 = self.sa30(x)
        
        return pga, pgv, sa03, sa10, sa30
    


class Model_cnn_for_embedding(nn.Module):
    def __init__(self):
        super(Model_cnn_for_embedding, self).__init__()
        self.model_chosen = 'nofeatures'
        self.trace_len =1000
        self.relu = nn.ReLU()
        if self.trace_len==1000:
            dim_in=12
        elif self.trace_len==2500:
            dim_in=24
        else:
            print("trace_len not valid")
        
        self.conv3 = nn.Conv2d(dim_in, 64, (39, 5), stride=(39, 5), padding=(19, 2))
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        
        self.dense1 = nn.Linear(128*dim_in, 128)

        self.graph_features_flattened = nn.Flatten()
        
        if self.model_chosen == 'nofeatures':
            self.merged_dense = nn.Linear(2048, 128)
        if self.model_chosen == 'main':
            self.merged_dense = nn.Linear(2048 + 78, 128)

        self.pga = nn.Linear(128, 39)
        self.pgv = nn.Linear(128, 39)
        self.sa03 = nn.Linear(128, 39)
        self.sa10 = nn.Linear(128, 39)
        self.sa30 = nn.Linear(128, 39)

    def forward(self, wav_input, graph_features, graph_input):
        model_chosen = self.model_chosen
        # wav_input=wav_input.permute(0,3,1,2)
        # print("in" ,wav_input.shape)
        # conv1_output = self.relu(self.conv1(wav_input))
        # print("conv1" ,conv1_output.shape)
        # conv2_output = self.relu(self.conv2(conv1_output))
        # print("conv2" ,conv2_output.shape)
        conv3_output = self.relu(self.conv3(wav_input))
        # print("conv3" ,conv3_output.shape)
        

        conv3_output_flattened = self.flatten(conv3_output)
        # print("conv3 flat" ,conv3_output_flattened.shape)
        
        conv3_output_flattened_dropout = self.dropout(conv3_output_flattened)
        # print("conv3 flat drop" ,conv3_output_flattened_dropout.shape)
        

        # graph_features_flattened = self.graph_features_flattened(graph_features)
        # print("graph_features_flattened" ,graph_features_flattened.shape)
        

        if model_chosen == 'nofeatures':
            merged = self.relu(self.merged_dense(conv3_output_flattened_dropout))
        if model_chosen == 'main':
            merged = torch.cat((conv3_output_flattened_dropout, graph_features_flattened), dim=1)
            merged = self.relu(self.merged_dense(merged))
        # print("merged" ,merged.shape)
        
        pga_output = self.pga(merged)
        pgv_output = self.pgv(merged)
        sa03_output = self.sa03(merged)
        sa10_output = self.sa10(merged)
        sa30_output = self.sa30(merged)

        return pga_output, pgv_output, sa03_output, sa10_output, sa30_output



class Model_gcn_for_embedding(nn.Module):
    def __init__(self):
        super(Model_gcn_for_embedding, self).__init__()
        self.model_chosen = 'nofeatures'
        self.relu = nn.ReLU()
        self.trace_len =1000
        if self.trace_len==1000:
            dim_in=12
        elif self.trace_len==2500:
            dim_in=24
        else:
            print("trace_len not valid")
         
        gcn_input_dim = 64 * ((39 // 4) // 39)  # Approximate reshaped width after conv layers

        self.gcn1 = GCNConv(gcn_input_dim + 2 if self.model_chosen == 'main' else gcn_input_dim, 64)
        self.gcn2 = GCNConv(64, 64)
        
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(2496, 128)
        self.pga = nn.Linear(128, 39)
        self.pgv = nn.Linear(128, 39)
        self.sa03 = nn.Linear(128, 39)
        self.sa10 = nn.Linear(128, 39)
        self.sa30 = nn.Linear(128, 39)
        
        
    def forward(self, wav_input, graph_features, graph_input):
        batch_size= wav_input.shape[0]
        x = wav_input.reshape(batch_size,39,-1)
        # print("x after reshape",x.shape)
        edge_index, edge_weight = dense_to_sparse(torch.Tensor(graph_input[0]))

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x = F.relu(self.gcn1(data.x,  data.edge_index, data.edge_attr))#graph_input))#
        # print("gcn1",x.shape)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x = torch.tanh(self.gcn2(data.x,  data.edge_index, data.edge_attr))#graph_input))#
        # print("gcn2",x.shape)
        
        # x = x.view(batch_size, -1)
        x = x.view(batch_size, -1)
        # print("x view",x.shape)
        
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        # print("fc1",x.shape)
        
        pga = self.pga(x)
        pgv = self.pgv(x)
        sa03 = self.sa03(x)
        sa10 = self.sa10(x)
        sa30 = self.sa30(x)
        
        return pga, pgv, sa03, sa10, sa30
    

