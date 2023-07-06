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
        dropout: float = 0.5,
    ):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(num_node_features, num_hidden_units)
        self.conv2 = GCNConv(num_hidden_units, num_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return torch.sigmoid(x)
