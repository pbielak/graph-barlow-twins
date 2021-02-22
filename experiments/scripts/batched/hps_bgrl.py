import json
import os
import sys
import yaml

import pandas as pd
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
from torch_geometric.data import Data
from torch_geometric.data import NeighborSampler
from tqdm import tqdm


from gssl import DATA_DIR
from gssl.augment import GraphAugmentor
from gssl.datasets import load_dataset
from gssl.utils import load_cls_from_str
from gssl.utils import seed

from GCL.models import BootstrapContrast
from GCL.losses import BootstrapLatent

from src.augmentation_grid import make_augmentation_params_grid, is_same
from src.bgrl import BatchedBGRL, compute_tau, train_batched, test


def evaluate_single_graph_batched_model(
    dataset_name: str,
    params: dict,
    augmentor: GraphAugmentor,
    batch_size: int,
):
    # Load data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, masks = load_dataset(name=dataset_name)

    # Build model
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
    train_loader = NeighborSampler(
        data.edge_index,
        node_idx=masks[0]["train"] if params["use_train_mask"] else None,
        sizes=[10,] * encoder.num_layers,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(os.environ.get("NUM_WORKERS", 0)),
    )

    for epoch in tqdm(range(params["total_epochs"]), desc="Epochs"):
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

    _, accs = test(
        model=model,
        data=data,
        masks=masks[0],
        use_pytorch_eval_model=params["use_pytorch_eval_model"],
        device=device,
    )

    return accs


def main():
    seed()

    # Read dataset name
    dataset_name = sys.argv[1]

    # Read params
    with open("experiments/configs/batched/hps_bgrl.yaml", "r") as fin:
        params = yaml.safe_load(fin)[dataset_name]

    metric_name = params["metric"]

    out_dir = os.path.join(DATA_DIR, f"batched/hps_bgrl/{dataset_name}/")
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, "log.csv")
    best_file = os.path.join(out_dir, "best.json")

    # Make hyperparameter space
    grid = make_augmentation_params_grid(
        p_x_min=params["p_x"]["min"],
        p_x_max=params["p_x"]["max"],
        p_x_step=params["p_x"]["step"],
        p_e_min=params["p_e"]["min"],
        p_e_max=params["p_e"]["max"],
        p_e_step=params["p_e"]["step"],
        grid_type=params["grid_type"],
    )

    records = []

    best_different = {
        bs: {metric_name: -1, "params": None}
        for bs in params["batch_sizes"]
    }
    best_same = {
        bs: {metric_name: -1, "params": None}
        for bs in params["batch_sizes"]
    }

    for batch_size in tqdm(params["batch_sizes"], desc="Batch sizes"):
        for augmentation_params in tqdm(grid, desc="Grid search"):
            augmentor = GraphAugmentor(**augmentation_params)

            if dataset_name == "PPI":  # Multiple graphs scenario
                raise NotImplemented()
            else:  # Single graph scenario
                metrics = evaluate_single_graph_batched_model(
                    dataset_name=dataset_name,
                    params=params,
                    augmentor=augmentor,
                    batch_size=batch_size,
                )

            records.append({
                "batch_size": batch_size,
                **augmentation_params,
                metric_name: metrics,
            })

            (
                pd.DataFrame.from_records(records)
                .to_csv(path_or_buf=log_file, index=False)
            )

            if is_same(augmentation_params):
                if metrics["test"] > best_same[batch_size][metric_name]:
                    best_same[batch_size][metric_name] = metrics["test"]
                    best_same[batch_size]["params"] = augmentation_params
            else:
                if metrics["test"] > best_different[batch_size][metric_name]:
                    best_different[batch_size][metric_name] = metrics["test"]
                    best_different[batch_size]["params"] = augmentation_params

            with open(best_file, "w") as fout:
                json.dump(
                    obj={"SAME": best_same, "DIFFERENT": best_different},
                    fp=fout,
                    indent=4,
                )


if __name__ == "__main__":
    main()
