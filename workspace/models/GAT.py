import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class GAT(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        num_hidden_units: int,
        num_heads: int = 1,
        dropout: float = 0.25,
    ) -> None:
        super(GAT, self).__init__()

        self.num_heads = num_heads
        self.num_hidden_units = num_hidden_units
        self.conv1 = GATConv(num_node_features, num_hidden_units * num_heads)
        self.conv2 = GATConv(num_hidden_units * num_heads, num_hidden_units, heads=1)
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
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index, edge_attr=edge_attr)
            if i != len(self.gnn_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        x = self.MLP(x)

        return torch.sigmoid(x)
