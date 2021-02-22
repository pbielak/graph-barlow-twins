import os
from typing import Dict, List, Tuple

from ogb.nodeproppred import PygNodePropPredDataset
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, WikiCS
from torch_geometric import transforms as T
from torch_geometric.utils import to_undirected

from gssl import DATA_DIR


def load_dataset(name: str) -> Tuple[Data, List[Dict[str, torch.Tensor]]]:
    ds_path = os.path.join(DATA_DIR, f"ssl/datasets/", name)
    feature_norm = T.NormalizeFeatures()
    create_masks = T.AddTrainValTestMask(
        split="train_rest",
        num_splits=20,
        num_val=0.1,
        num_test=0.8,
    )

    if name == "WikiCS":
        data = WikiCS(
            root=ds_path,
            transform=feature_norm,
        )[0]
    elif name == "Amazon-CS":
        data = Amazon(
            root=ds_path,
            name="computers",
            transform=feature_norm,
            pre_transform=create_masks,
        )[0]
    elif name == "Amazon-Photo":
        data = Amazon(
            root=ds_path,
            name="photo",
            transform=feature_norm,
            pre_transform=create_masks,
        )[0]
    elif name == "Coauthor-CS":
        data = Coauthor(
            root=ds_path,
            name="cs",
            transform=feature_norm,
            pre_transform=create_masks,
        )[0]
    elif name == "Coauthor-Physics":
        data = Coauthor(
            root=ds_path,
            name="physics",
            transform=feature_norm,
            pre_transform=create_masks,
        )[0]
    elif name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(
            root=ds_path,
            name="ogbn-arxiv",
        )
        split_idx = dataset.get_idx_split()

        data = dataset[0]

        data.train_mask = torch.zeros((data.num_nodes,), dtype=torch.bool)
        data.train_mask[split_idx["train"]] = True

        data.val_mask = torch.zeros((data.num_nodes,), dtype=torch.bool)
        data.val_mask[split_idx["valid"]] = True

        data.test_mask = torch.zeros((data.num_nodes,), dtype=torch.bool)
        data.test_mask[split_idx["test"]] = True

        data.y = data.y.squeeze(dim=-1)

        data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if name == "ogbn-arxiv":
        masks = [
            {
                "train": data.train_mask,
                "val": data.val_mask,
                "test": data.test_mask,
            }
        ]
    else:
        masks = [
            {
                "train": data.train_mask[:, i],
                "val": data.val_mask[:, i],
                "test": (
                    data.test_mask
                    if name == "WikiCS"
                    else data.test_mask[:, i]
                ),
            }
            for i in range(20)
        ]

    return data, masks
