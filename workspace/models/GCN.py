import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        num_hidden_units: int,
        dropout: float = 0.25,
    ):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(num_node_features, num_hidden_units)
        self.conv2 = GCNConv(num_hidden_units, num_hidden_units)
        self.gnn_layers = torch.nn.ModuleList([self.conv1, self.conv2])
        self.dropout = torch.nn.Dropout(dropout)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(num_hidden_units, num_hidden_units),
            self.dropout,
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_units, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_weight=edge_attr)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.MLP(x)

        return torch.sigmoid(x)
