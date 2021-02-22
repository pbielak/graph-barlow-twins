import json
import os

from gssl import DATA_DIR


def print_metrics(path: str, log_intervals: list):
    with open(path, "r") as fin:
        metrics = json.load(fin)

    for epoch in log_intervals:
        val_mean = metrics[f"val_acc_{epoch}_mean"]
        val_std = metrics[f"val_acc_{epoch}_std"]

        test_mean = metrics[f"test_acc_{epoch}_mean"]
        test_std = metrics[f"test_acc_{epoch}_std"]

        print(
            f"Epoch {epoch} => Val: {val_mean:.2f} +/- {val_std:.2f}, Test: {test_mean:.2f} +/- {test_std:.2f}"
        )


def main():
    print("--- G-BT ---")
    print_metrics(
        path=os.path.join(DATA_DIR, "batched/train/ogbn-products/512/metrics.json"),
        log_intervals=[i * 10 for i in range(11)],
    )

    print("--- BGRL ---")
    print_metrics(
        path=os.path.join(DATA_DIR, "batched/train_bgrl/ogbn-products/512/metrics.json"),
        log_intervals=[*[i * 10 for i in range(10)], 99]
    )


if __name__ == "__main__":
    main()

