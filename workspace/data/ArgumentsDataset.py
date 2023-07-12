import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


class ArgumentsDataset(InMemoryDataset):
    def __init__(self, root, test_size=0.2, transform=None, pre_transform=None) -> None:
        self.test_size = test_size
        super(ArgumentsDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list:
        return [
            "edge_index.npy",
            "edge_attributes.npy",
            "bert_embeddings.npy",
            "node_labels.npy",
        ]

    @property
    def processed_file_names(self) -> list:
        return ["data.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        # Load data
        edge_index = torch.from_numpy(np.load(self.raw_paths[0])).long()
        edge_attr = torch.from_numpy(np.load(self.raw_paths[1])).float()
        node_emb = torch.from_numpy(np.load(self.raw_paths[2])).float()
        labels = torch.from_numpy(np.load(self.raw_paths[3])).long()

        num_nodes = node_emb.shape[0]

        # Create mask
        indices = torch.randperm(num_nodes)
        train_mask = indices[: int((1 - self.test_size) * num_nodes)]
        test_mask = indices[int((1 - self.test_size) * num_nodes) :]

        mask_train = torch.zeros(num_nodes, dtype=bool)
        mask_test = torch.zeros(num_nodes, dtype=bool)

        mask_train[train_mask] = True
        mask_test[test_mask] = True

        data = Data(
            x=node_emb,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels,
            train_mask=mask_train,
            test_mask=mask_test,
        )

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
