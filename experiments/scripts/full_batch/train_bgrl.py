import json
import os
import sys
import yaml

import numpy as np
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from tqdm.auto import tqdm

from GCL.models import BootstrapContrast
from GCL.losses import BootstrapLatent

from gssl import DATA_DIR
from gssl.augment import GraphAugmentor
from gssl.datasets import load_dataset, load_ppi
from gssl.full_batch.multiple_graph_model import FullBatchMultipleGraphModel
from gssl.utils import seed, plot_vectors

from src.bgrl import compute_tau, BGRL, TwoLayerGCN, test


def train(
    model: BGRL,
    contrast_model: BootstrapContrast,
    optimizer: torch.optim.AdamW,
    scheduler: LinearWarmupCosineAnnealingLR,
    data: Data,
    device: torch.device,
    tau: float,
):
    model.train()
    contrast_model.train()

    optimizer.zero_grad()

    _, _, _h1_pred, _h2_pred, _h1_target, _h2_target = model(data.to(device))
    loss = contrast_model(
        h1_pred=_h1_pred, h2_pred=_h2_pred,
        h1_target=_h1_target.detach(), h2_target=_h2_target.detach(),
    )

    loss.backward()
    optimizer.step()
    scheduler.step()
    model.update_target_encoder(tau)

    return loss.item()


def evaluate_single_graph_full_batch_model(
    dataset_name: str,
    params: dict,
    augmentor: GraphAugmentor,
    logger: SummaryWriter,
    retrain_idx: int,
):
    # Load data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, masks = load_dataset(name=dataset_name)

    # Build BGRL model
    encoder = TwoLayerGCN(
        in_dim=data.num_node_features,
        hidden_dim=params["hidden_dim"],
        out_dim=params["emb_dim"],
    )
    model = BGRL(
        encoder=encoder,
        augmentor=augmentor,
        out_dim=params["emb_dim"],
        pred_dim=params["pred_dim"],
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
    for epoch in tqdm(range(params["total_epochs"]), desc="Epoch"):
        loss = train(
            model=model,
            contrast_model=contrast_model,
            optimizer=optimizer,
            scheduler=scheduler,
            data=data,
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
                masks=masks[0 if dataset_name == "ogbn-arxiv" else retrain_idx],
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


#def evaluate_multiple_graph_full_batch_model(
#    params: dict,
#    augmentor: GraphAugmentor,
#    logger: SummaryWriter,
#):
#    train_data, val_data, test_data = load_ppi()
#
#    encoder = load_cls_from_str(params["encoder_cls"])(
#        in_dim=train_data[0].num_node_features,
#        out_dim=params["emb_dim"],
#    )
#
#    model = FullBatchMultipleGraphModel(
#        encoder=encoder,
#        augmentor=augmentor,
#        lr_base=params["lr_base"],
#        total_epochs=params["total_epochs"],
#        warmup_epochs=params["warmup_epochs"],
#    )
#
#    logs = model.fit(
#        data={"train": train_data, "val": val_data, "test": test_data},
#        logger=logger,
#        log_interval=params["log_interval"],
#    )
#
#    return model, logs


def main():
    seed()

    # Read dataset name
    dataset_name = sys.argv[1]

    # Read params
    with open("experiments/configs/full_batch/train_bgrl.yaml", "r") as fin:
        params = yaml.safe_load(fin)[dataset_name]

    if dataset_name == "PPI":
        metric_full_name = "f1"
        metric_short_name = "f1"
    else:
        metric_full_name = "accuracies"
        metric_short_name = "acc"

    outs_dir = os.path.join(DATA_DIR, f"full_batch/train_bgrl/{dataset_name}/")

    log_epochs = None
    train_metrics = []
    val_metrics = []
    test_metrics = []

    #for i in tqdm(range(20), desc="Splits"):
    for i in tqdm(range(5), desc="Splits"):
        logger = SummaryWriter(log_dir=os.path.join(outs_dir, f"logs/{i}"))

        augmentor = GraphAugmentor(
            p_x_1=params["p_f_1"],
            p_x_2=params["p_f_2"],
            p_e_1=params["p_e_1"],
            p_e_2=params["p_e_2"],
        )

        if dataset_name == "PPI":  # Multiple graphs scenario
            raise NotImplemented()
            # model, logs = evaluate_multiple_graph_full_batch_model(
            #     params=params,
            #     augmentor=augmentor,
            #     logger=logger,
            # )
        else:  # Single graph scenario
            logs = evaluate_single_graph_full_batch_model(
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
