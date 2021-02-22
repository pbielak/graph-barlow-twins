import json
import os

from gssl import DATA_DIR


def get_params(dataset: str, method: str) -> dict:
    params = {}

    # Metric
    if dataset == "PPI":
        params["metric_name"] = "f1"
    else:
        params["metric_name"] = "acc"

    # Log intervals
    if method == "GBT":
        if dataset == "ogbn-arxiv":
            params["log_intervals"] = [0, 100, 200, 300, 400, 500]
        elif dataset == "PPI":
            params["log_intervals"] = [0, 100, 200, 300, 400, 500]
        else:
            params["log_intervals"] = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1_000]

        # Path
        params["path"] = "train"
    elif method == "BGRL":
        params["log_intervals"] = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9999]
        # Path
        params["path"] = "train_bgrl"
    else:
        raise RuntimeError(f"Unknown method: {method}")

    return params


def main():
    datasets = [
        "WikiCS", 
        "Amazon-CS", "Amazon-Photo", 
        "Coauthor-CS", "Coauthor-Physics", 
        "ogbn-arxiv",
        "PPI",
    ]

    for dataset in datasets:
        print(f"Dataset: {dataset}")

        for method in ("GBT", "BGRL"):
            params = get_params(dataset=dataset, method=method)

            metrics_file = os.path.join(
                DATA_DIR, f"full_batch/{params['path']}/{dataset}/metrics.json"
            )

            if not os.path.exists(metrics_file):
                print(f"{method} => Not found")
                continue
            
            with open(metrics_file, "r") as fin:
                metrics = json.load(fin)

            best_test_mean = -1
            best_test_std = -1
            best_test_num_epochs = None

            for num_epochs in params["log_intervals"]:
                mean = metrics[f"test_{params['metric_name']}_{num_epochs}_mean"]
                std = metrics[f"test_{params['metric_name']}_{num_epochs}_std"]

                if mean > best_test_mean:
                    best_test_mean = mean
                    best_test_std = std
                    best_test_num_epochs = num_epochs

            if dataset in ("ogbn-arxiv", "PPI"):
                val_mean = metrics[f"val_{params['metric_name']}_{best_test_num_epochs}_mean"]
                val_std = metrics[f"val_{params['metric_name']}_{best_test_num_epochs}_std"]
                print(
                    f"{method} => "
                    f"Val {params['metric_name'].title()}: {val_mean:.2f} +/- {val_std:.2f} "
                    f"Test {params['metric_name'].title()}: {best_test_mean:.2f} +/- {best_test_std:.2f} "
                    f"(at epoch {best_test_num_epochs})"
                )
            else:
                print(
                    f"{method} => "
                    f"Test {params['metric_name'].title()}: {best_test_mean:.2f} +/- {best_test_std:.2f} "
                    f"(at epoch {best_test_num_epochs})"
                )

            if method == "BGRL":   # Print for 1k epochs
                test_1000_mean = metrics[f"test_{params['metric_name']}_1000_mean"]
                test_1000_std = metrics[f"test_{params['metric_name']}_1000_std"]
                print(
                    f"{method} => "
                    f"Test {params['metric_name'].title()}: {test_1000_mean:.2f} +/- {test_1000_std:.2f} "
                    "(at epoch 1000)"
                )


if __name__ == "__main__":
    main()

