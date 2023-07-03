import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class ArgumentsDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ArgumentsDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # Return the names of the raw data files.
        return ['edge_index.npy', 'edge_attributes.npy', 'bert_embeddings.npy', 'labels.npy']

    @property
    def processed_file_names(self):
        # Return the names of the processed data files.
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read raw data files and construct `Data` instances.

        edge_index = np.load(self.raw_paths[0])
        edge_attr = np.load(self.raw_paths[1])
        node_emb = np.load(self.raw_paths[2])

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        node_emb = torch.tensor(node_emb, dtype=torch.float)
        labels = torch.tensor(np.load(self.raw_paths[3]), dtype=torch.long)

        data = Data(x=node_emb, edge_index=edge_index, edge_attr=edge_attr, y=labels)

        torch.save(self.collate([data]), self.processed_paths[0])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_paths[idx])
        return data
