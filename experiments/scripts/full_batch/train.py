import json
import os
import sys
import yaml

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


from gssl import DATA_DIR
from gssl.augment import GraphAugmentor
from gssl.datasets import load_dataset, load_ppi
from gssl.full_batch.model import FullBatchModel
from gssl.full_batch.multiple_graph_model import FullBatchMultipleGraphModel
from gssl.utils import load_cls_from_str
from gssl.utils import seed


def evaluate_single_graph_full_batch_model(
    dataset_name: str,
    params: dict,
    augmentor: GraphAugmentor,
    logger: SummaryWriter,
    retrain_idx: int,
):
    data, masks = load_dataset(name=dataset_name)

    encoder = load_cls_from_str(params["encoder_cls"])(
        in_dim=data.num_node_features,
        out_dim=params["emb_dim"],
    )

    model = FullBatchModel(
        encoder=encoder,
        augmentor=augmentor,
        lr_base=params["lr_base"],
        total_epochs=params["total_epochs"],
        warmup_epochs=params["warmup_epochs"],
        use_pytorch_eval_model=params["use_pytorch_eval_model"],
    )

    logs = model.fit(
        data=data,
        logger=logger,
        log_interval=params["log_interval"],
        masks=masks[
            0 if dataset_name == "ogbn-arxiv"
            else retrain_idx
        ],
    )

    return model, logs


def evaluate_multiple_graph_full_batch_model(
    params: dict,
    augmentor: GraphAugmentor,
    logger: SummaryWriter,
):
    train_data, val_data, test_data = load_ppi()

    encoder = load_cls_from_str(params["encoder_cls"])(
        in_dim=train_data[0].num_node_features,
        out_dim=params["emb_dim"],
    )

    model = FullBatchMultipleGraphModel(
        encoder=encoder,
        augmentor=augmentor,
        lr_base=params["lr_base"],
        total_epochs=params["total_epochs"],
        warmup_epochs=params["warmup_epochs"],
    )

    logs = model.fit(
        data={"train": train_data, "val": val_data, "test": test_data},
        logger=logger,
        log_interval=params["log_interval"],
    )

    return model, logs


def main():
    seed()

    # Read dataset name
    dataset_name = sys.argv[1]

    # Read params
    with open("experiments/configs/full_batch/train.yaml", "r") as fin:
        params = yaml.safe_load(fin)[dataset_name]

    with open(f"data/full_batch/hps/{dataset_name}/best.json", "r") as fin:
        aug_params = json.load(fp=fin)

    if dataset_name == "PPI":
        metric_full_name = "f1"
        metric_short_name = "f1"
    else:
        metric_full_name = "accuracies"
        metric_short_name = "acc"

    outs_dir = os.path.join(DATA_DIR, f"full_batch/train/{dataset_name}/")

    emb_dir = os.path.join(outs_dir, "embeddings/")
    os.makedirs(emb_dir, exist_ok=True)

    model_dir = os.path.join(outs_dir, "models/")
    os.makedirs(model_dir, exist_ok=True)

    log_epochs = None
    train_metrics = []
    val_metrics = []
    test_metrics = []

    for i in tqdm(range(20), desc="Splits"):
        logger = SummaryWriter(log_dir=os.path.join(outs_dir, f"logs/{i}"))

        augmentor = GraphAugmentor(
            p_x_1=aug_params["SAME"]["params"]["p_x_1"],
            p_e_1=aug_params["SAME"]["params"]["p_e_1"],
        )

        if dataset_name == "PPI":  # Multiple graphs scenario
            model, logs = evaluate_multiple_graph_full_batch_model(
                params=params,
                augmentor=augmentor,
                logger=logger,
            )
        else:  # Single graph scenario
            model, logs = evaluate_single_graph_full_batch_model(
                dataset_name=dataset_name,
                params=params,
                augmentor=augmentor,
                logger=logger,
                retrain_idx=i,
            )

        log_epochs = logs["log_epoch"]
        train_metrics.append(logs[f"train_{metric_full_name}"])
        val_metrics.append(logs[f"val_{metric_full_name}"])
        test_metrics.append(logs[f"test_{metric_full_name}"])

        # Save latent vectors (embeddings)
        for epoch, z in zip(logs["log_epoch"], logs["z"]):
            torch.save(obj=z, f=os.path.join(emb_dir, f"{i}_epoch{epoch}.pt"))

        # Save model
        torch.save(obj=model, f=os.path.join(model_dir, f"{i}.pt"))

    # Save metadata
    metadata = {
        "epochs": log_epochs,
        f"train_{metric_full_name}": train_metrics,
        f"val_{metric_full_name}": val_metrics,
        f"test_{metric_full_name}": test_metrics,
    }
    metadata_path = os.path.join(outs_dir, "metadata.json")

    with open(metadata_path, "w") as fout:
        json.dump(obj=metadata, fp=fout, indent=4)

    # Save metrics
    metrics = {}

    for idx in range(len(log_epochs)):
        epoch = log_epochs[idx]
        mean_val = np.mean([m[idx] for m in val_metrics])
        std_val = np.std([m[idx] for m in val_metrics], ddof=1)

        mean_test = np.mean([m[idx] for m in test_metrics])
        std_test = np.std([m[idx] for m in test_metrics], ddof=1)

        metrics.update({
            f"val_{metric_short_name}_{epoch}_mean": mean_val * 100.0,
            f"val_{metric_short_name}_{epoch}_std": std_val * 100.0,

            f"test_{metric_short_name}_{epoch}_mean": mean_test * 100.0,
            f"test_{metric_short_name}_{epoch}_std": std_test * 100.0,
        })

    metrics_path = os.path.join(outs_dir, "metrics.json")

    with open(metrics_path, "w") as fout:
        json.dump(obj=metrics, fp=fout, indent=4)


if __name__ == "__main__":
    main()
