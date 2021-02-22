import os
import sys
import yaml

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


from gssl import DATA_DIR
from gssl.inductive.datasets import load_ppi
from gssl.inductive.model import Model
from gssl.inductive.tasks import evaluate_node_classification
from gssl.utils import seed


def main():
    seed()

    # Read dataset name
    dataset_name = "PPI"
    loss_name = sys.argv[1]

    # Read params
    with open(
        "experiments_ssl/configs/hyperparameter_search_inductive_ppi.yaml",
        "r"
    ) as fin:
        params = yaml.safe_load(fin)[loss_name][dataset_name]

    train_data, val_data, test_data = load_ppi()

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

    records = []

    for p_x, p_e in tqdm(grid, desc="Grid search"):
        model = Model(
            feature_dim=train_data[0].x.size(-1),
            emb_dim=params["emb_dim"],
            loss_name=loss_name,
            p_x=p_x,
            p_e=p_e,
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

        f1 = evaluate_node_classification(
            z_train=z_train, y_train=y_train,
            z_val=z_val, y_val=y_val,
            z_test=z_test, y_test=y_test,
        )

        records.append({
            "p_x": p_x,
            "p_e": p_e,
            "f1": f1,
        })

        (
            pd.DataFrame.from_records(records)
            .to_csv(path_or_buf=hps_file, index=False)
        )


if __name__ == "__main__":
    main()
