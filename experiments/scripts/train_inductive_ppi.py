import json
import os
import sys
import yaml

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


from gssl import DATA_DIR
from gssl.inductive.datasets import load_ppi
from gssl.inductive.model import Model
from gssl.utils import seed


def main():
    seed()

    # Read dataset name
    dataset_name = "PPI"
    loss_name = sys.argv[1]

    # Read params
    with open("experiments_ssl/configs/train_inductive_ppi.yaml", "r") as fin:
        params = yaml.safe_load(fin)[loss_name][dataset_name]

    train_data, val_data, test_data = load_ppi()

    outs_dir = os.path.join(
        DATA_DIR, f"ssl/{loss_name}/{dataset_name}/"
    )

    emb_dir = os.path.join(outs_dir, "embeddings/")
    os.makedirs(emb_dir, exist_ok=True)

    model_dir = os.path.join(outs_dir, "models/")
    os.makedirs(model_dir, exist_ok=True)

    log_epochs = None
    train_f1s = []
    val_f1s = []
    test_f1s = []

    for i in tqdm(range(20), desc="Splits"):
        logger = SummaryWriter(log_dir=os.path.join(outs_dir, f"logs/{i}"))

        model = Model(
            feature_dim=train_data[0].x.size(-1),
            emb_dim=params["emb_dim"],
            loss_name=loss_name,
            p_x=params["p_x"],
            p_e=params["p_e"],
            lr_base=params["lr_base"],
            total_epochs=params["total_epochs"],
            warmup_epochs=params["warmup_epochs"],
        )

        logs = model.fit(
            data={"train": train_data, "val": val_data, "test": test_data},
            logger=logger,
            log_interval=params["log_interval"],
        )

        log_epochs = logs["log_epoch"]
        train_f1s.append(logs["train_f1"])
        val_f1s.append(logs["val_f1"])
        test_f1s.append(logs["test_f1"])

        # Save latent vectors (embeddings)
        for epoch, z in zip(logs["log_epoch"], logs["z"]):
            torch.save(obj=z, f=os.path.join(emb_dir, f"{i}_epoch{epoch}.pt"))

        # Save model
        torch.save(obj=model, f=os.path.join(model_dir, f"{i}.pt"))

    # Save metadata
    metadata = {
        "epochs": log_epochs,
        "train_f1s": train_f1s,
        "val_f1s": val_f1s,
        "test_f1s": test_f1s,
    }
    metadata_path = os.path.join(outs_dir, "metadata.json")

    with open(metadata_path, "w") as fout:
        json.dump(obj=metadata, fp=fout, indent=4)

    # Save metrics
    metrics = {}

    for idx in range(len(log_epochs)):
        epoch = log_epochs[idx]
        mean_val_f1 = np.mean([f1[idx] for f1 in val_f1s])
        std_val_f1 = np.std([f1[idx] for f1 in val_f1s], ddof=1)

        mean_test_f1 = np.mean([f1[idx] for f1 in test_f1s])
        std_test_f1 = np.std([f1[idx] for f1 in test_f1s], ddof=1)

        metrics.update({
            f"val_f1_{epoch}_mean": mean_val_f1 * 100.0,
            f"val_f1_{epoch}_std": std_val_f1 * 100.0,

            f"test_f1_{epoch}_mean": mean_test_f1 * 100.0,
            f"test_f1_{epoch}_std": std_test_f1 * 100.0,
        })

    metrics_path = os.path.join(outs_dir, "metrics.json")

    with open(metrics_path, "w") as fout:
        json.dump(obj=metrics, fp=fout, indent=4)


if __name__ == "__main__":
    main()
