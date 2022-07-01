"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.

Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import HeteroGraphConv
from dgl.nn.pytorch import GraphConv
from torch.nn import init
from torch.nn import LSTM
class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, u, v):
        emb_u = self.u_embeddings(u)
        emb_v = self.v_embeddings(v)
        # emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        # neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        # neg_score = torch.clamp(neg_score, max=10, min=-10)
        # neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score)

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = HeteroGraphConv({
            rel: GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.LSTM_layers = nn.LSTM(input_size=out_size*layer_num_heads,hidden_size=64,num_layers=1,batch_first=True)
    def forward(self, g, h,sample):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:

                # self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                #         g, meta_path)
                self.g = g.to('cpu')
                self.new_g  = dgl.sampling.RandomWalkNeighborSampler(self.g,termination_prob=0.5,num_neighbors=100,num_random_walks=10,num_traversals=5,metapath=meta_path)
                self._cached_coalesced_graph[meta_path] = self.new_g(sample).to('cuda: 0')
        # print(len(self.meta_paths))
        for i, meta_path in enumerate(self.meta_paths):
            # print(i)
            new_g = self._cached_coalesced_graph[meta_path]

            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
            # meta_semantic_embeddings = self.gat_layers[i](new_g, h).flatten(1)
            # meta_emb = torch.zeros((meta_semantic_embeddings.shape[0], 4,meta_semantic_embeddings.shape[1])).cuda()
            # new_g = dgl.sampling.random_walk(self.g, sample, metapath=meta_path)
            # seq = new_g[0]
            # typ = new_g[1]
            # mask = torch.where(typ == 4)
            # seq_mat = seq[:, mask[0]]
            # emb_features = meta_semantic_embeddings
            # for j, node in enumerate(sample):
            #     node_emb = torch.zeros((4, meta_semantic_embeddings.shape[1]))
            #     for k, seq in enumerate(seq_mat[j]):
            #         if seq != -1:
            #             node_emb[k] = emb_features[seq]
            #     meta_emb[node] = node_emb
            # out_embeddings,_ = self.LSTM_layers(meta_emb,None)
            # out_embeddings = out_embeddings.permute(1,0,2)
            # semantic_embeddings.append(out_embeddings[-1])
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)
class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))

        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h,sample):
        for gnn in self.layers:
            h = gnn(g, h,sample)
        return self.predict(h)
