import json
import os

import numpy as np
import torch
from tqdm import tqdm

from gssl import DATA_DIR
from gssl.datasets import load_dataset
from gssl.tasks import evaluate_node_classification_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data, masks = load_dataset("ogbn-products")

    val_metrics = []
    test_metrics = []
    for retrain in tqdm(range(5), desc="Retrain"):
            z = data.x
            acc = evaluate_node_classification_acc(
                z=z, data=data, masks=masks[0], use_pytorch=True,
            )

            val_metrics.append(acc["val"] * 100.0)
            test_metrics.append(acc["test"] * 100.0)

    val_mean = np.mean(val_metrics)
    val_std = np.std(val_metrics, ddof=1)

    test_mean = np.mean(test_metrics)
    test_std = np.std(test_metrics, ddof=1)
    print(
        f"Val: {val_mean:.2f} +/- {val_std:.2f}\n"
        f"Test: {test_mean:.2f} +/- {test_std:.2f}"
    )

    metrics = {
        "val_mean": val_mean,
        "val_std": val_std,
        "test_mean": test_mean,
        "test_std": test_std,
    }

    out_path = os.path.join(DATA_DIR, "products/features.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as fout:
        json.dump(obj=metrics, fp=fout, indent=4)
            

if __name__ == "__main__":
    main()
