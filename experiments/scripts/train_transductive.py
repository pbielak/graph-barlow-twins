import json
import os
import sys
import yaml

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


from gssl import DATA_DIR
from gssl.datasets import load_dataset
from gssl.transductive_model import Model
from gssl.transductive_model_arxiv import ArxivModel
from gssl.utils import seed


def main():
    seed()

    # Read dataset name
    dataset_name = sys.argv[1]
    loss_name = sys.argv[2]

    # Read params
    with open("experiments_ssl/configs/train_transductive.yaml", "r") as fin:
        params = yaml.safe_load(fin)[loss_name][dataset_name]

    data, masks = load_dataset(name=dataset_name)

    outs_dir = os.path.join(
        DATA_DIR, f"ssl/{loss_name}/{dataset_name}/"
    )

    emb_dir = os.path.join(outs_dir, "embeddings/")
    os.makedirs(emb_dir, exist_ok=True)

    model_dir = os.path.join(outs_dir, "models/")
    os.makedirs(model_dir, exist_ok=True)

    # Which model to use
    if dataset_name == "ogbn-arxiv":
        model_cls = ArxivModel
    else:
        model_cls = Model

    log_epochs = None
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []

    for i in tqdm(range(20), desc="Splits"):
        logger = SummaryWriter(log_dir=os.path.join(outs_dir, f"logs/{i}"))

        model = model_cls(
            feature_dim=data.x.size(-1),
            emb_dim=params["emb_dim"],
            loss_name=loss_name,
            p_x=params["p_x"],
            p_e=params["p_e"],
            lr_base=params["lr_base"],
            total_epochs=params["total_epochs"],
            warmup_epochs=params["warmup_epochs"],
        )

        logs = model.fit(
            data=data,
            logger=logger,
            log_interval=params["log_interval"],
            masks=masks[
                0 if dataset_name == "ogbn-arxiv"
                else i
            ],
        )

        log_epochs = logs["log_epoch"]
        train_accuracies.append(logs["train_accuracies"])
        val_accuracies.append(logs["val_accuracies"])
        test_accuracies.append(logs["test_accuracies"])

        # Save latent vectors (embeddings)
        for epoch, z in zip(logs["log_epoch"], logs["z"]):
            torch.save(obj=z, f=os.path.join(emb_dir, f"{i}_epoch{epoch}.pt"))

        # Save model
        torch.save(obj=model, f=os.path.join(model_dir, f"{i}.pt"))

    # Save metadata
    metadata = {
        "epochs": log_epochs,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "test_accuracies": test_accuracies,
    }
    metadata_path = os.path.join(outs_dir, "metadata.json")

    with open(metadata_path, "w") as fout:
        json.dump(obj=metadata, fp=fout, indent=4)

    # Save metrics
    metrics = {}

    for idx in range(len(log_epochs)):
        epoch = log_epochs[idx]
        mean_val_acc = np.mean([acc[idx] for acc in val_accuracies])
        std_val_acc = np.std([acc[idx] for acc in val_accuracies], ddof=1)

        mean_test_acc = np.mean([acc[idx] for acc in test_accuracies])
        std_test_acc = np.std([acc[idx] for acc in test_accuracies], ddof=1)

        metrics.update({
            f"val_acc_{epoch}_mean": mean_val_acc * 100.0,
            f"val_acc_{epoch}_std": std_val_acc * 100.0,

            f"test_acc_{epoch}_mean": mean_test_acc * 100.0,
            f"test_acc_{epoch}_std": std_test_acc * 100.0,
        })

    metrics_path = os.path.join(outs_dir, "metrics.json")

    with open(metrics_path, "w") as fout:
        json.dump(obj=metrics, fp=fout, indent=4)


if __name__ == "__main__":
    main()
