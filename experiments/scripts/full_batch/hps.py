import json
import os
import sys

import pandas as pd
import torch
from tqdm.auto import tqdm
import yaml

from gssl import DATA_DIR
from gssl.augment import GraphAugmentor
from gssl.datasets import load_dataset, load_ppi
from gssl.full_batch.model import FullBatchModel
from gssl.full_batch.multiple_graph_model import FullBatchMultipleGraphModel
from gssl.tasks import (
    evaluate_node_classification_acc,
    evaluate_node_classification_multilabel_f1,
)
from gssl.utils import load_cls_from_str, seed

from src.augmentation_grid import make_augmentation_params_grid, is_same


def evaluate_single_graph_full_batch_model(
    dataset_name: str,
    params: dict,
    augmentor: GraphAugmentor,
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

    model.fit(data=data)

    z = model.predict(data=data)
    accuracy = evaluate_node_classification_acc(
        z=z, data=data, masks=masks[0],
        use_pytorch=params["use_pytorch_eval_model"],
    )

    return accuracy


def evaluate_multiple_graph_full_batch_model(
    params: dict,
    augmentor: GraphAugmentor,
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
    with open("experiments/configs/full_batch/hps.yaml", "r") as fin:
        params = yaml.safe_load(fin)[dataset_name]

    metric_name = params["metric"]

    out_dir = os.path.join(DATA_DIR, f"full_batch/hps/{dataset_name}/")
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

    best_different = {metric_name: -1, "params": None}
    best_same = {metric_name: -1, "params": None}

    for augmentation_params in tqdm(grid, desc="Grid search"):
        augmentor = GraphAugmentor(**augmentation_params)

        if dataset_name == "PPI":  # Multiple graphs scenario
            metrics = evaluate_multiple_graph_full_batch_model(
                params=params,
                augmentor=augmentor,
            )
        else:  # Single graph scenario
            metrics = evaluate_single_graph_full_batch_model(
                dataset_name=dataset_name,
                params=params,
                augmentor=augmentor,
            )

        records.append({**augmentation_params, metric_name: metrics})

        (
            pd.DataFrame.from_records(records)
            .to_csv(path_or_buf=log_file, index=False)
        )

        if is_same(augmentation_params):
            if metrics["test"] > best_same[metric_name]:
                best_same[metric_name] = metrics["test"]
                best_same["params"] = augmentation_params
        else:
            if metrics["test"] > best_different[metric_name]:
                best_different[metric_name] = metrics["test"]
                best_different["params"] = augmentation_params

        with open(best_file, "w") as fout:
            json.dump(
                obj={"SAME": best_same, "DIFFERENT": best_different},
                fp=fout,
                indent=4,
            )


if __name__ == "__main__":
    main()
