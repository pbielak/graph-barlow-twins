import json
import os
import sys
import yaml

import pandas as pd
import torch
from tqdm.auto import tqdm


from gssl import DATA_DIR
from gssl.augment import GraphAugmentor
from gssl.batched.model import BatchedModel
from gssl.batched.multiple_graph_model import BatchedMultipleGraphModel
from gssl.datasets import load_dataset, load_ppi
from gssl.tasks import (
    evaluate_node_classification_acc,
    evaluate_node_classification_multilabel_f1,
)
from gssl.utils import load_cls_from_str
from gssl.utils import seed

from src.augmentation_grid import make_augmentation_params_grid, is_same


def evaluate_single_graph_batched_model(
    dataset_name: str,
    params: dict,
    augmentor: GraphAugmentor,
    batch_size: int,
):
    data, masks = load_dataset(name=dataset_name)

    encoder = load_cls_from_str(params["encoder_cls"])(
        in_dim=data.num_node_features,
        out_dim=params["emb_dim"],
    )

    model = BatchedModel(
        encoder=encoder,
        augmentor=augmentor,
        lr_base=params["lr_base"],
        total_epochs=params["total_epochs"],
        warmup_epochs=params["warmup_epochs"],
        batch_size=batch_size,
        inference_batch_size=params["inference_batch_size"],
        use_pytorch_eval_model=params["use_pytorch_eval_model"],
        use_train_mask=params["use_train_mask"],
    )

    model.fit(data=data, masks=masks[0])

    z = model.predict(data=data)
    accuracy = evaluate_node_classification_acc(
        z=z, data=data, masks=masks[0],
        use_pytorch=params["use_pytorch_eval_model"],
    )

    return accuracy


def evaluate_multiple_graph_batched_model(
    params: dict,
    augmentor: GraphAugmentor,
    batch_size: int,
):
    train_data, val_data, test_data = load_ppi()

    encoder = load_cls_from_str(params["encoder_cls"])(
        in_dim=train_data[0].num_node_features,
        out_dim=params["emb_dim"],
    )

    model = BatchedMultipleGraphModel(
        encoder=encoder,
        augmentor=augmentor,
        lr_base=params["lr_base"],
        total_epochs=params["total_epochs"],
        warmup_epochs=params["warmup_epochs"],
        batch_size=batch_size,
        inference_batch_size=params["inference_batch_size"],
    )

    model.fit(data={"train": train_data})

    z_train = model.predict(data=train_data)
    z_val = model.predict(data=val_data)
    z_test = model.predict(data=test_data)

    y_train = torch.cat([trd.y for trd in train_data], dim=0)
    y_val = torch.cat([vd.y for vd in val_data], dim=0)
    y_test = torch.cat([ted.y for ted in test_data], dim=0)

    f1 = evaluate_node_classification_multilabel_f1(
        z_train=z_train, y_train=y_train,
        z_val=z_val, y_val=y_val,
        z_test=z_test, y_test=y_test,
    )

    return f1


def main():
    seed()

    # Read dataset name
    dataset_name = sys.argv[1]

    # Read params
    with open("experiments/configs/batched/hps.yaml", "r") as fin:
        params = yaml.safe_load(fin)[dataset_name]

    metric_name = params["metric"]

    out_dir = os.path.join(DATA_DIR, f"batched/hps/{dataset_name}/")
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
                metrics = evaluate_multiple_graph_batched_model(
                    params=params,
                    augmentor=augmentor,
                    batch_size=batch_size,
                )
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
