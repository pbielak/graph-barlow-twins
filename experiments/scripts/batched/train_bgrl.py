import json
import os
import sys
import yaml

import numpy as np
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import NeighborSampler
from tqdm.auto import tqdm

from GCL.models import BootstrapContrast
from GCL.losses import BootstrapLatent

from gssl import DATA_DIR
from gssl.augment import GraphAugmentor
from gssl.datasets import load_dataset
from gssl.utils import seed, load_cls_from_str, plot_vectors

from src.bgrl import BatchedBGRL, compute_tau, train_batched, test


def evaluate_single_graph_batched_model(
    dataset_name: str,
    params: dict,
    augmentor: GraphAugmentor,
    batch_size: int,
    logger: SummaryWriter,
):
    # Load data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, masks = load_dataset(name=dataset_name)

    # Build BGRL model
    encoder = load_cls_from_str(params["encoder_cls"])(
        in_dim=data.num_node_features,
        out_dim=params["emb_dim"],
    )
    model = BatchedBGRL(
        encoder=encoder,
        augmentor=augmentor,
        out_dim=params["emb_dim"],
        pred_dim=params["pred_dim"],
        inference_batch_size=params["inference_batch_size"],
    ).to(device)

    contrast_model = BootstrapContrast(
        loss=BootstrapLatent(),
        mode="L2L",
    ).to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=params["lr_base"],
        weight_decay=1e-5,
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=params["warmup_epochs"],
        max_epochs=params["total_epochs"],
    )

    # Train and evaluate
    logs = {
        "log_epoch": [],
        "train_accuracies": [],
        "val_accuracies": [],
        "test_accuracies": [],
    }
    train_loader = NeighborSampler(
        data.edge_index,
        node_idx=masks[0]["train"] if params["use_train_mask"] else None,
        sizes=[10,] * encoder.num_layers,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(os.environ.get("NUM_WORKERS", 0)),
    )

    for epoch in tqdm(range(params["total_epochs"]), desc="Epoch"):
        loss = train_batched(
            model=model,
            contrast_model=contrast_model,
            optimizer=optimizer,
            scheduler=scheduler,
            data=data,
            loader=train_loader,
            device=device,
            tau=compute_tau(epoch, params["total_epochs"]),
        )

        logger.add_scalar("Loss", loss, epoch)

        if (
            epoch % params["log_interval"] == 0
            or epoch == params["total_epochs"] - 1
        ):
            z, accs = test(
                model=model,
                data=data,
                masks=masks[0],
                use_pytorch_eval_model=params["use_pytorch_eval_model"],
                device=device,
            )

            logger.add_figure(
                "latent",
                plot_vectors(z, labels=data.y.cpu()),
                epoch,
            )
            logger.add_scalar("acc/train", accs["train"], epoch)
            logger.add_scalar("acc/val", accs["val"], epoch)
            logger.add_scalar("acc/test", accs["test"], epoch)

            logs["log_epoch"].append(epoch)
            logs["train_accuracies"].append(accs["train"])
            logs["val_accuracies"].append(accs["val"])
            logs["test_accuracies"].append(accs["test"])

            logger.add_scalar("norm", z.norm(dim=1).mean(), epoch)

    return logs


def main():
    seed()

    # Read dataset name
    dataset_name = sys.argv[1]

    # Read params
    with open("experiments/configs/batched/train_bgrl.yaml", "r") as fin:
        params = yaml.safe_load(fin)[dataset_name]

    with open(f"data/batched/hps_bgrl/{dataset_name}/best.json", "r") as fin:
        aug_params = json.load(fp=fin)

    if dataset_name == "PPI":
        metric_full_name = "f1"
        metric_short_name = "f1"
    else:
        metric_full_name = "accuracies"
        metric_short_name = "acc"

    for batch_size in tqdm(params["batch_sizes"], desc="Batch sizes"):
        outs_dir = os.path.join(
            DATA_DIR, f"batched/train_bgrl/{dataset_name}/{batch_size}/",
        )

        log_epochs = None
        train_metrics = []
        val_metrics = []
        test_metrics = []

        for i in tqdm(range(params["num_splits"]), desc="Splits"):
            logger = SummaryWriter(log_dir=os.path.join(outs_dir, f"logs/{i}"))

            augmentor = GraphAugmentor(
                p_x_1=aug_params["SAME"][str(batch_size)]["params"]["p_x_1"],
                p_e_1=aug_params["SAME"][str(batch_size)]["params"]["p_e_1"],
            )

            if dataset_name == "PPI":  # Multiple graphs scenario
                raise NotImplemented()
            else:  # Single graph scenario
                logs = evaluate_single_graph_batched_model(
                    dataset_name=dataset_name,
                    params=params,
                    augmentor=augmentor,
                    batch_size=batch_size,
                    logger=logger,
                )

            log_epochs = logs["log_epoch"]
            train_metrics.append(logs[f"train_{metric_full_name}"])
            val_metrics.append(logs[f"val_{metric_full_name}"])
            test_metrics.append(logs[f"test_{metric_full_name}"])

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
