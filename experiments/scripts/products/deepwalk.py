import argparse
import json
import os

import numpy as np
import torch
from torch_geometric.nn import Node2Vec
from tqdm import tqdm

from gssl import DATA_DIR
from gssl.datasets import load_dataset
from gssl.tasks import evaluate_node_classification_acc



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_dim = 128
    walk_length = 40
    context_size = 20
    walks_per_node = 10
    batch_size = 512
    lr = 1e-2
    epochs = 5

    data, masks = load_dataset("ogbn-products")

    metrics = {
        "deepwalk": {"val": [], "test": []},
        "deepwalk+features": {"val": [], "test": []},
    }

    for _ in tqdm(range(5), desc="Retrain"):
        model = Node2Vec(
            edge_index=data.edge_index, 
            embedding_dim=embedding_dim, 
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            p=1,
            q=1,
            sparse=True,
        ).to(device)

        loader = model.loader(
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(os.environ.get("NUM_WORKERS", 4)),
        )
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

        model.train()
        for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
            for pos_rw, neg_rw in tqdm(loader, desc="Batches", leave=False):
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()

        z = model.embedding.weight.data.cpu().detach()
        acc = evaluate_node_classification_acc(
            z=z, data=data, masks=masks[0], use_pytorch=True,
        )

        metrics["deepwalk"]["val"].append(acc["val"])
        metrics["deepwalk"]["test"].append(acc["test"])

        acc_f = evaluate_node_classification_acc(
            z=torch.cat([z, data.x], dim=-1),
            data=data,
            masks=masks[0],
            use_pytorch=True,
        )

        metrics["deepwalk+features"]["val"].append(acc_f["val"])
        metrics["deepwalk+features"]["test"].append(acc_f["test"])

        # Save during training
        out_metrics = {}
        for key in ("deepwalk", "deepwalk+features"):
            out_metrics[key] = {
                "val_mean": np.mean(metrics[key]["val"]),
                "val_std": np.std(metrics[key]["val"], ddof=1),
                "val_all": metrics[key]["val"],
                "test_mean": np.mean(metrics[key]["test"]),
                "test_std": np.std(metrics[key]["test"], ddof=1),
                "test_all": metrics[key]["test"],
            }

        metrics_dir = os.path.join(DATA_DIR, "products/")
        os.makedirs(metrics_dir, exist_ok=True)

        with open(os.path.join(metrics_dir, "deepwalk.json"), "w") as fout:
            json.dump(obj=out_metrics["deepwalk"], fp=fout, indent=4)

        with open(os.path.join(metrics_dir, "deepwalk_features.json"), "w") as fout:
            json.dump(obj=out_metrics["deepwalk+features"], fp=fout, indent=4)
            

if __name__ == "__main__":
    main()
