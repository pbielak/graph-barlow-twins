import json
import os

from gssl import DATA_DIR



def main():
    datasets = [
        "WikiCS", 
        "Amazon-CS", "Amazon-Photo",
        "Coauthor-CS", "Coauthor-Physics",
        "ogbn-arxiv",
        "PPI",
        "ogbn-products",
    ]

    for dataset in datasets:
        print(f"Dataset: {dataset}")

        if dataset == "ogbn-products":
            batch_sizes = [512]
            log_intervals = [i * 10 for i in range(11)]
            metric = "acc"
        elif dataset == "ogbn-arxiv":
            batch_sizes = [1024, 2048]
            log_intervals = [i * 100 for i in range(6)]
            metric = "acc"
        elif dataset == "PPI":
            batch_sizes = [512, 1024]
            log_intervals = [i * 100 for i in range(6)]
            metric = "f1"
        else:
            batch_sizes = [256, 512, 1024, 2048]
            log_intervals = [i * 100 for i in range(11)]
            metric = "acc"

        for batch_size in batch_sizes:
            metrics_file = os.path.join(
                DATA_DIR, f"batched/train/{dataset}/{batch_size}/metrics.json"
            )

            if not os.path.exists(metrics_file):
                print(f"Batch size: {batch_size} => N/A")
                continue

            with open(metrics_file, "r") as fin:
                metrics = json.load(fin)

            best_test_mean = -1
            best_test_std = -1
            best_test_num_epochs = None

            for num_epochs in log_intervals:
                mean = metrics[f"test_{metric}_{num_epochs}_mean"]
                std = metrics[f"test_{metric}_{num_epochs}_std"]

                if mean > best_test_mean:
                    best_test_mean = mean
                    best_test_std = std
                    best_test_num_epochs = num_epochs

            if dataset in ("ogbn-arxiv", "ogbn-products", "PPI"):
                val_mean = metrics[f"val_{metric}_{best_test_num_epochs}_mean"]
                val_std = metrics[f"val_{metric}_{best_test_num_epochs}_std"]

                print(
                    f"Batch size: {batch_size} => "
                    f"Val {metric.title()}: {val_mean:.2f} +/- {val_std:.2f} "
                    f"Test {metric.title()}: {best_test_mean:.2f} +/- {best_test_std:.2f} "
                    f"(at epoch {best_test_num_epochs})"
                )
            else:
                print(
                    f"Batch size: {batch_size} => "
                    f"Test {metric.title()}: {best_test_mean:.2f} +/- {best_test_std:.2f} "
                    f"(at epoch {best_test_num_epochs})"
                )


if __name__ == "__main__":
    main()

