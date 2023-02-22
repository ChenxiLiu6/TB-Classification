import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from utils import *
#from torch.nn import GCNConv
#from torch.nn.pool.topk_pool import topk, filter_adj
from torch.nn import Parameter


def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    """
    if use_cuda == 1:
        rtn = nn.Parameter(torch.Tensor(*shape).cuda())
    else:
        """
    rtn = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(rtn)  # using a uniform distribution
    return rtn

def get_att(x, W, emb_dim, batch_size=32):
    temp = torch.mean(x, 1).view((batch_size, 1, -1))  # (batch_size, 1, D)->([742, 1, 128])
    h_avg = torch.tanh(torch.matmul(temp, W)) # shape: ([742, 1, 128])
    att = torch.bmm(x, h_avg.transpose(2, 1)) # shape: ([742, 29, 1])
    output = torch.bmm(att.transpose(2, 1), x) # shape: ([742, 1, 128])

    return output

class GraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, aggr='mean', bias=True):
        assert aggr in ['add', 'mean', 'max']
        super(GraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr

        self.weight = Parameter(torch.Tensor(in_channels, out_channels)) # (429, 128)
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        self.lin.reset_parameters()


    def forward(self, x, adj, mask=None):
        x = x.unsqueeze(0) if x.dim() == 2 else x          # x shape: ([742, 29, 429])
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj  # adj shape: ([742, 29, 29])
        B, N, _ = adj.size()

        out = torch.matmul(adj, x)            # out shape: ([742, 29, 429])
        out = torch.matmul(out, self.weight)  # weight shape: ([429, 128]), out shape: ([742, 29, 128])


        if self.aggr == 'mean':
            out = out / adj.sum(dim=-1, keepdim=True).clamp(min=1) # shape: [742, 29, 128]
            # print("mean out: ", out.shape)
        elif self.aggr == 'max':
            out = out.max(dim=-1)[0]

        # print("lin(x) shape: ", self.lin(x).shape) # shape: ([742/186, 29, 128])
        out = out + self.lin(x)

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class Adaptive_Pooling_Layer(nn.Module):
    """ Memory_Pooling_Layer introduced in the paper 'MEMORY-BASED GRAPH NETWORKS' by Amir Hosein Khasahmadi, etc"""

    ### This layer is for downsampling a node set from N_input to N_output
    ### Input: [B,N_input,Dim_input]
    ###         B:batch size, N_input: number of input nodes, Dim_input: dimension of features of input nodes
    ### Output:[B,N_output,Dim_output]
    ###         B:batch size, N_output: number of output nodes, Dim_output: dimension of features of output nodes

    def __init__(self, Heads, Dim_input, N_output, Dim_output):
        """
            Heads: number of memory heads
            N_input : number of nodes in input node set
            Dim_input: number of feature dimension of input nodes
            N_output : number of the downsampled output nodes
            Dim_output: number of feature dimension of output nodes
        """
        super(Adaptive_Pooling_Layer, self).__init__()
        self.Heads = Heads
        self.Dim_input = Dim_input
        self.N_output = N_output
        self.Dim_output = Dim_output
        #self.use_cuda = use_cuda

        #if self.use_cuda == 1:
            #self.device = torch.device("cuda:0,1")
        #else:
        self.device = torch.device("cpu")

        # Randomly initialize centroids
        self.centroids = nn.Parameter(2 * torch.rand(self.Heads, self.N_output, Dim_input) - 1)   # shape: (4, 1, 128)
        self.centroids.requires_grad = True

        hiden_channels = self.Heads # 4
        """
        if self.use_cuda == 1:
            self.input2centroids_weight = torch.nn.Parameter(
                torch.zeros(hiden_channels, 1).float().to(self.device), requires_grad=True)
            self.input2centroids_bias = torch.nn.Parameter(
                torch.zeros(hiden_channels).float().to(self.device), requires_grad=True)
        else:
            """
        self.input2centroids_weight = torch.nn.Parameter(
            torch.zeros(hiden_channels, 1).float(), requires_grad=True) # shape: (4, 1)
        self.input2centroids_bias = torch.nn.Parameter(
            torch.zeros(hiden_channels).float(), requires_grad=True)    # shape: (4)

        self.input2centroids_ = nn.Sequential(nn.Linear(hiden_channels, self.Heads * self.N_output), nn.ReLU())

        self.memory_aggregation = nn.Conv2d(self.Heads, 1, [1, 1])

        self.dim_feat_transformation = nn.Linear(self.Dim_input, self.Dim_output)

        self.similarity_compute = torch.nn.CosineSimilarity(dim=4, eps=1e-6)

        self.relu = nn.ReLU()

        self.emb_dim = Dim_input # 128
        self.W_0 = glorot([self.emb_dim, self.emb_dim])  # (128, 128)

    def forward(self, node_set, adj, zero_tensor=torch.tensor([0])):
        """
            node_set: Input node set in form of [batch_size, N_input, Dim_input]
            adj: adjacency matrix for node set x in form of [batch_size, N_input, N_input]
            zero_tensor: zero_tensor of size [1]

            (1): new_node_set = LRelu(C*node_set*W)
            (2): C = softmax(pixel_level_conv(C_heads))
            (3): C_heads = t-distribution(node_set, centroids)
            (4): W is a trainable linear transformation
        """
        node_set_input = node_set
        batch_size, self.N_input, _ = node_set.size()  # N_input = 29, shape:(742, 29, 128), batch_size: 742 / 186

        #batch_centroids = torch.mean(node_set,dim=1,keepdim=True)

        batch_centroids = get_att(node_set, self.W_0, self.emb_dim, batch_size=batch_size) # shape: (742, 1, 128) #self.emb_dim,
        batch_centroids = batch_centroids.permute(0, 2, 1) # shape: (742, 128, 1)

        batch_centroids = torch.relu(
            torch.nn.functional.linear(batch_centroids, self.input2centroids_weight, self.input2centroids_bias)) # [742, 128, 4]

        batch_centroids = self.input2centroids_(batch_centroids) # shape: [742, 128, 4]

        batch_centroids = batch_centroids.permute(0, 2, 1).view(node_set.size()[0], self.Heads, self.N_output,
                                                                self.Dim_input) # shape: ([742, 4, 1, 128])

        # From initial node set [batch_size, N_input, Dim_input] to size [batch_size, Heads, N_output, N_input, Dim_input]
        node_set = torch.unsqueeze(node_set, 1).repeat(1, batch_centroids.shape[1], 1, 1) # shape: [742, 4, 29, 128]
        node_set = torch.unsqueeze(node_set, 2).repeat(1, 1, batch_centroids.shape[2], 1, 1) # shape: ([742, 4, 1, 29, 128])
        # Broadcast centroids to the same size as that of node set [batch_size, Heads, N_output, N_input, Dim_input]
        batch_centroids = torch.unsqueeze(batch_centroids, 3).repeat(1, 1, 1, node_set.shape[3], 1) # shape: ([742, 4, 1, 29, 128])

        # Compute the distance between original node set to centroids
        # [batch_size, Heads, N_output, N_input]
        C_heads = self.similarity_compute(node_set, batch_centroids) # shape: ([742, 4, 1, 29])

        normalizer = torch.unsqueeze(torch.sum(C_heads, 2), 2) # ([742, 4, 1, 29])
        C_heads = C_heads / (normalizer + 1e-10)

        # Apply pixel-level convolution and softmax to C_heads
        # Get C: [batch_size, N_output, N_input]
        C = self.memory_aggregation(C_heads) # shape: [742, 1, 1, 29]
        # C = torch.softmax(C,1)

        C = C.squeeze(1) # shape: ([742, 1, 29])


        # 3. Compute Assignment matrix: X_pool, A_pool
        # [batch_size, N_output, N_input] * [batch_size, N_input, Dim_input] --> [batch_size, N_output, Dim_input]
        new_node_set = torch.matmul(C, node_set_input) # shape: ([742, 1, 128])
        # [batch_size, N_output, Dim_input] * [batch_size, Dim_input, Dim_output] --> [batch_size, N_output, Dim_output]
        new_node_set = self.dim_feat_transformation(new_node_set) # shape:([742, 1, 128])


        # [batch_size, N_output, N_input] * [batch_size, N_input, N_input] --> [batch_size, N_output, N_input]
        q_adj = torch.matmul(C, adj) # shape: [742, 1, 29]
        # [batch_size, N_output, N_input] * [batch_size, N_input, N_output] --> [batch_size, N_output, N_output]
        new_adj = self.relu(torch.matmul(q_adj, C.transpose(1, 2))) # ([742, 1, 1])

        return new_node_set, new_adj

class Memory_Pooling_Layer(nn.Module):
    """ Memory_Pooling_Layer introduced in the paper 'MEMORY-BASED GRAPH NETWORKS' by Amir Hosein Khasahmadi, etc"""

    ### This layer is for downsampling a node set from N_input to N_output
    ### Input: [B,N_input,Dim_input]
    ###         B:batch size, N_input: number of input nodes, Dim_input: dimension of features of input nodes
    ### Output:[B,N_output,Dim_output]
    ###         B:batch size, N_output: number of output nodes, Dim_output: dimension of features of output nodes

    def __init__(self, Heads, Dim_input, N_output, Dim_output, Tau=1):
        """
            Heads: number of memory heads
            N_input : number of nodes in input node set
            Dim_input: number of feature dimension of input nodes
            N_output : number of the downsampled output nodes
            Dim_output: number of feature dimension of output nodes
            Tau: parameter for the student t-distribution mentioned in the paper
        """
        super(Memory_Pooling_Layer, self).__init__()
        self.Heads = Heads
        self.Dim_input = Dim_input
        self.N_output = N_output
        self.Dim_output = Dim_output
        self.Tau = Tau

        # Randomly initialize centroids
        self.centroids = nn.Parameter(2 * torch.rand(self.Heads, self.N_output, Dim_input) - 1)
        self.centroids.requires_grad = True

        self.memory_aggregation = nn.Conv2d(self.Heads, 1, [1, 1])
        self.dim_feat_transformation = nn.Linear(self.Dim_input, self.Dim_output)
        self.lrelu = nn.LeakyReLU()

    def forward(self, node_set, adj, zero_tensor=torch.tensor([0])):
        """
            node_set: Input node set in form of [batch_size, N_input, Dim_input]
            adj: adjacency matrix for node set x in form of [batch_size, N_input, N_input]
            zero_tensor: zero_tensor of size [1]

            (1): new_node_set = LRelu(C*node_set*W)
            (2): C = softmax(pixel_level_conv(C_heads))
            (3): C_heads = t-distribution(node_set, centroids)
            (4): W is a trainable linear transformation
        """
        node_set_input = node_set
        _, self.N_input, _ = node_set.size()

        """
            With (1)(2)(3)(4) we calculate new_node_set
        """

        # Copy centroids and repeat it in batch
        ## [h,N_output,Dim_input] --> [batch_size,Heads,N_output,Dim_input]
        batch_centroids = torch.unsqueeze(self.centroids, 0). \
            repeat(node_set.shape[0], 1, 1, 1)
        # From initial node set [batch_size, N_input, Dim_input] to size [batch_size, Heads, N_output, N_input, Dim_input]
        node_set = torch.unsqueeze(node_set, 1).repeat(1, batch_centroids.shape[1], 1, 1)
        node_set = torch.unsqueeze(node_set, 2).repeat(1, 1, batch_centroids.shape[2], 1, 1)
        # Broadcast centroids to the same size as that of node set [batch_size, Heads, N_output, N_input, Dim_input]
        batch_centroids = torch.unsqueeze(batch_centroids, 3).repeat(1, 1, 1, node_set.shape[3], 1)
        # Compute the distance between original node set to centroids
        # [batch_size, Heads, N_output, N_input]

        dist = torch.sum(torch.abs(node_set - batch_centroids) ** 2, 4)

        # Compute the Matrix C_heads : [batch_size, Heads, N_output, N_input]
        C_heads = torch.pow((1 + dist / self.Tau), -(self.Tau + 1) / 2)

        normalizer = torch.unsqueeze(torch.sum(C_heads, 2), 2)
        C_heads = C_heads / normalizer

        # Apply pixel-level convolution and softmax to C_heads
        # Get C: [batch_size, N_output, N_input]
        C = self.memory_aggregation(C_heads)
        C = torch.softmax(C, 1)

        C = C.squeeze(1)

        # [batch_size, N_output, N_input] * [batch_size, N_input, Dim_input] --> [batch_size, N_output, Dim_input]
        new_node_set = torch.matmul(C, node_set_input)
        # [batch_size, N_output, Dim_input] * [batch_size, Dim_input, Dim_output] --> [batch_size, N_output, Dim_output]
        new_node_set = self.dim_feat_transformation(new_node_set)
        new_node_set = self.lrelu(new_node_set)

        """
            Calculate new_adj
        """
        # [batch_size, N_output, N_input] * [batch_size, N_input, N_input] --> [batch_size, N_output, N_input]
        q_adj = torch.matmul(C, adj)

        # [batch_size, N_output, N_input] * [batch_size, N_input, N_output] --> [batch_size, N_output, N_output]
        new_adj = torch.matmul(q_adj, C.transpose(1, 2))

        return new_node_set, new_adj