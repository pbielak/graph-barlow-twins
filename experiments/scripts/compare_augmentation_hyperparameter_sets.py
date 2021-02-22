import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from gssl import DATA_DIR


def load_results(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath_or_buffer=os.path.join(
            DATA_DIR,
            f"full_batch/hps/{dataset}/log.csv",
        ),
        converters={"accuracy": lambda s: json.loads(s.replace("'", "\""))}
    )
    df["dataset"] = dataset

    df["test_accuracy"] = df["accuracy"].apply(lambda x: x["test"] * 100.0)

    df["p1"] = df[["p_x_1", "p_e_1"]].apply(
        lambda x: (x["p_x_1"], x["p_e_1"]),
        axis=1,
    )
    df["p2"] = df[["p_x_2", "p_e_2"]].apply(
        lambda x: (x["p_x_2"], x["p_e_2"]),
        axis=1,
    )

    df["param_sets"] = ""
    df.loc[df["p1"] == df["p2"], "param_sets"] = "Same"
    df.loc[df["p1"] != df["p2"], "param_sets"] = "Different"

    return df[["dataset", "p1", "p2", "param_sets", "test_accuracy"]]


def main():
    datasets = [
        "WikiCS",
        "Amazon-CS",
        "Amazon-Photo",
        "Coauthor-CS",
        "Coauthor-Physics",
    ]

    fig, axs = plt.subplots(figsize=(10, 3), ncols=len(datasets))
    plt.rcParams.update({'axes.titlesize': 14})

    for i, (dataset, ax) in enumerate(zip(datasets, axs)):
        df = load_results(dataset=dataset)

        # Plot scores for same and different augmentation parameter sets
        sns.boxplot(
            data=df,
            x="param_sets",
            y="test_accuracy",
            ax=ax,
        )

        # Find best scores
        eq = df[df["param_sets"] == "Same"]
        eq_best = eq[eq["test_accuracy"] == eq["test_accuracy"].max()]

        eq_best_value = eq_best["test_accuracy"].values[0]

        neq = df[df["param_sets"] == "Different"]
        neq_best = neq[neq["test_accuracy"] == neq["test_accuracy"].max()]

        neq_best_value = neq_best["test_accuracy"].values[0]

        ax.set(
            title=dataset,
            xlabel="",
            ylabel="Test accuracy [%]" if i == 0 else "",
        )
        ax.xaxis.set_tick_params(labelsize=12)

        for x_tick, val in zip(ax.get_xticks(), (eq_best_value, neq_best_value)):
            ax.text(
                x_tick, 1.0005 * val,
                f"{val:.2f}",
                horizontalalignment="center",
                size=14,
            )

        if i == 0:
            ax.yaxis.label.set_size(14)

        low, high = ax.get_ylim()
        ax.set(ylim=(low, high + 0.5))

    plt.tight_layout()
    plt.savefig("data/augmentation_hyperparameter_plot.png")


if __name__ == "__main__":
    main()
