#
import torch
from torch import nn

import torch_geometric as pyg
from torch_geometric.nn import MessagePassing

__all__ = [
    "MeshGraphNet",
    "MeshGraphLayer",
]

#======================================================================#
# MeshGraphNets
# https://medium.com/stanford-cs224w/learning-mesh-based-flow-simulations-on-graph-networks-44983679cf2d
#======================================================================#
class MeshGraphNet(nn.Module):
    def __init__(self, ci_node, ci_edge, co, hidden_dim, num_layers, **kwargs):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(ci_node, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(ci_edge, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.processor = nn.ModuleList()
        for _ in range(num_layers):
            layer = MeshGraphLayer(hidden_dim, hidden_dim, **kwargs)
            self.processor.append(layer)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, co),
        )

        return

    def forward(self, data):
        xn, xe = data.x, data.edge_attr
        edge_index = data.edge_index

        xn = self.node_encoder(xn)
        xe = self.edge_encoder(xe)

        for layer in self.processor:
            xn, xe = layer(xn, xe, edge_index)

        xn = self.decoder(xn)

        return xn
#

class MeshGraphLayer(MessagePassing):
    def __init__(self, ci, co, **kwargs):
        super().__init__(**kwargs)

        self.node_mlp = nn.Sequential(
            nn.Linear(2 * ci, co),
            nn.ReLU(),
            nn.Linear(co, co),
            nn.LayerNorm(co),
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * ci, co),
            nn.ReLU(),
            nn.Linear(co, co),
            nn.LayerNorm(co),
        )

        return

    def forward(self, xn, xe, edge_index):

        msg, ye = self.propagate(edge_index, x=xn, edge_attr=xe, dim_size=xn.size(0))

        yn = torch.cat([msg, xn], dim=1) # cat messages with node features
        yn = self.node_mlp(yn)           # apply node MLP
        yn = yn + xn                     # residual connection

        return yn, ye

    def message(self, x_i, x_j, edge_attr):

        xe = edge_attr
        ye = torch.cat([x_i, x_j, xe], dim=1) # cat node and edge features
        ye = self.edge_mlp(ye)                # apply edge MLP
        ye = xe + ye                          # residual connection

        return ye

    def aggregate(self, ye, edge_index, dim_size=None):

        import torch_scatter

        node_dim = 0
        msg = torch_scatter.scatter(
            ye, edge_index[0,:], dim=node_dim,
            dim_size=dim_size, reduce='sum',
        )

        return msg, ye
#======================================================================#
#
