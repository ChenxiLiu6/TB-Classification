import torch
import torch.nn as nn
import torch.nn.functional as F

# from utils import config_dataset, corr_matrix
from MTPool_layer import GraphConv, Adaptive_Pooling_Layer, Memory_Pooling_Layer


class ConvPool(nn.Module):
    def __init__(self):
        super(ConvPool, self).__init__()
        # conv layer
        self.conv1 = nn.Conv1d(29, 16, 20, 1) # output: 742, 16, 128
        self.conv2 = nn.Conv1d(16, 4, 33, 1) # output: 742, 4, 32
        self.pool = nn.MaxPool1d(2, 2)

        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)           # (None, 16, 128)
        x = self.relu(self.pool(x)) # (None, 16, 64)
        x = self.conv2(x)           # (None, 4, 32)
        x = self.relu(self.pool(x)) # (None, 4, 16)

        x = x.view(x.size(0),-1)   # (None, 64)
        x = self.relu(self.fc1(x)) # (None, 16)
        x = self.sigmoid(self.fc2(x)) #(None, 1)

        return x

class MTPool(nn.Module):
    def __init__(self, graph_method, relation_method, pooling_method):
        super(MTPool, self).__init__()
        """
        @ graph_method: GNN
        @ relation_method: corr
        @ pooling_method: CoSimPool
        """
        self.graph_method=graph_method        # GNN
        self.relation_method=relation_method  # corr
        self.pooling_method=pooling_method    # CoSimPool

        self.num_nodes=29           # number of variables
        self.feature_dim=147        # timeseries length

        # use cpu or gpu
        self.device = torch.device('cpu')

        # -------------------------------------- 2. CNN Temporal Convolution ------------------------------------------#
        # CNN to extract feature
        kernel_ = [3, 5, 7]
        channel = 1
        self.c1 = nn.Conv2d(1, channel, kernel_size=(1, kernel_[0]), stride=1)
        self.c2 = nn.Conv2d(1, channel, kernel_size=(1, kernel_[1]), stride=1)
        self.c3 = nn.Conv2d(1, channel, kernel_size=(1, kernel_[2]), stride=1)

        d = (len(kernel_) * (self.feature_dim) - sum(kernel_) + len(kernel_)) * channel   # 429
        # -------------------------------------- 3. Spatial-Temporal Modeling ------------------------------------------#
        # GNN to extract feature
        self.hid = 128


        if self.graph_method == 'GNN':
            self.gnn = GraphConv(d, self.hid)

        if self.pooling_method == "CoSimPool":
            adaptive_pooling_layers = []

            ap = Adaptive_Pooling_Layer(Heads=4, Dim_input=self.hid, N_output=1, Dim_output=self.hid)
            adaptive_pooling_layers.append(ap)
            """
            ######### new add ap layers
            ap1 = Adaptive_Pooling_Layer(Heads=4, Dim_input=self.hid, N_output=16, Dim_output=self.hid)
            adaptive_pooling_layers.append(ap1)
            ap3 = Adaptive_Pooling_Layer(Heads=4, Dim_input=self.hid, N_output=1, Dim_output=self.hid)
            adaptive_pooling_layers.append(ap3)
            """

            ########
            self.ap = nn.ModuleList(adaptive_pooling_layers) # append a given module in a list

        elif self.pooling_method == 'MemPool':
            memory_pooling_layers = []
            mp = Memory_Pooling_Layer(Heads=4, Dim_input=self.hid, N_output=1, Dim_output=self.hid)
            memory_pooling_layers.append(mp)
            self.mp = nn.ModuleList(memory_pooling_layers)

        self.mlp = nn.Sequential(
            nn.Linear(self.hid, self.hid),
            nn.PReLU(),
            nn.Linear(self.hid, 1),
            nn.Sigmoid()
        )

        self.cnn_act = nn.PReLU() #nn.PReLU()
        self.gnn_act = nn.PReLU() #nn.PReLU()

        self.batch_norm_cnn = nn.BatchNorm1d(self.num_nodes)
        self.batch_norm_gnn = nn.BatchNorm1d(self.num_nodes) # 29
        self.batch_norm_mlp = nn.BatchNorm1d(self.hid)       # 128

        # (4) conv pool
        #self.p1 = nn.Conv1d(29, 1, 1, stride=1)


    def forward(self, X, corr):
        # x is N * L, where L is the time-series feature dimension

        # (1) Graph Adjacency Matrix
        self.A = corr # shape: (bs, 29, 29)

        # Process: input -> CNN -> Graph Adjacency Matrix -> GNN -> Pooling -> MLP -> output
        # (2) CNN to extract feature -> Part 2: Temporal Convolution
        a1 = self.c1(X.unsqueeze(1)).reshape(X.shape[0], X.shape[1], -1) # a1 shape:  torch.Size([742, 29, 145])
        a2 = self.c2(X.unsqueeze(1)).reshape(X.shape[0], X.shape[1], -1) # a2 shape:  torch.Size([742, 29, 143])
        a3 = self.c3(X.unsqueeze(1)).reshape(X.shape[0], X.shape[1], -1) # a3 shape:  torch.Size([742, 29, 141])
        x = self.cnn_act(torch.cat([a1, a2, a3], 2)) # x shape:  torch.Size([742, 29, 429])
        x = self.batch_norm_cnn(x)                   # ✅final x shape:  torch.Size([742, 29, 429])


        # (3) GNN
        if self.graph_method == 'GNN':
            #x = self.gnn_act(self.gnn(x,self.A))
            x = self.gnn(x, self.A)
            x = nn.PReLU()(x)
            x = x.squeeze(1)
            x = self.batch_norm_gnn(x)  # shape: [742/186, 29, 128] (bs, 29, 128)

        # (4.1) Pooling
        if self.pooling_method == 'CoSimPool':
            A = self.A
            for layer in self.ap:
                x, A = layer(x, A)

        # x shape: ([742, 1, 128])
        x = x.squeeze(1) # shape: (bs, 128)
        x = self.batch_norm_mlp(x) # shape: [742, 128]
        y = self.mlp(x) # shape:(bs, 1)

        return y

