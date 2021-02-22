import os
import sys
import yaml

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


from gssl import DATA_DIR
from gssl.datasets import load_dataset
from gssl.transductive_model import Model
from gssl.transductive_model_arxiv import ArxivModel
from gssl.tasks import evaluate_node_classification
from gssl.utils import seed


def main():
    seed()

    # Read dataset name
    dataset_name = sys.argv[1]
    loss_name = sys.argv[2]

    # Read params
    with open(
        "experiments_ssl/configs/hyperparameter_search_transductive.yaml",
        "r"
    ) as fin:
        params = yaml.safe_load(fin)[loss_name][dataset_name]

    data, masks = load_dataset(name=dataset_name)

    hps_file = os.path.join(
        DATA_DIR, f"ssl/{loss_name}/{dataset_name}/hps.csv"
    )
    os.makedirs(os.path.dirname(hps_file), exist_ok=True)

    # Make hyperparameter space
    grid = [
        (p_x, p_e)
        for p_x in np.arange(
            params["p_x"]["min"],
            params["p_x"]["max"] + params["p_x"]["step"],
            params["p_x"]["step"],
        )
        for p_e in np.arange(
            params["p_e"]["min"],
            params["p_e"]["max"] + params["p_e"]["step"],
            params["p_e"]["step"],
        )
    ]

    # Which model to use
    if dataset_name == "ogbn-arxiv":
        model_cls = ArxivModel
    else:
        model_cls = Model

    records = []

    for p_x, p_e in tqdm(grid, desc="Grid search"):
        model = model_cls(
            feature_dim=data.x.size(-1),
            emb_dim=params["emb_dim"],
            loss_name=loss_name,
            p_x=p_x,
            p_e=p_e,
            lr_base=params["lr_base"],
            total_epochs=params["total_epochs"],
            warmup_epochs=params["warmup_epochs"],
        )

        model.fit(data=data)

        z = model.predict(data=data)
        accuracy = evaluate_node_classification(
            z=z, data=data, masks=masks[0],
            use_pytorch=(dataset_name == "ogbn-arxiv"),
        )

        records.append({
            "p_x": p_x,
            "p_e": p_e,
            "accuracy": accuracy,
        })

        (
            pd.DataFrame.from_records(records)
            .to_csv(path_or_buf=hps_file, index=False)
        )


if __name__ == "__main__":
    main()
