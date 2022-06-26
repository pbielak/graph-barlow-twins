import matplotlib.pyplot as plt
import pandas as pd


def main():
    data = pd.read_csv("data/lambda_ablation_results.csv")
    data = data.sort_values(by="name")

    fig, ax = plt.subplots(figsize=(15, 5))
    plt.rcParams.update({'axes.titlesize': 14})

    lambda_values = data["name"]
    mean_acc = data["mean"] * 100.
    std_acc = data["std"] * 100.

    x = range(len(lambda_values))

    # Plot accuracy results
    ax.plot(x, mean_acc, linestyle="--", marker="o")
    ax.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, alpha=0.3)

    # Mark results for "lambda = 1/d"
    default_lambda_index = list(lambda_values).index(1 / 256)  # NOTE: hard-coded
    ax.axvline(
        x=default_lambda_index,
        linestyle="-.",
        color="r",
    )
    ax.text(
        x=0.28,
        y=0.48,
        s="$\lambda = \\frac{1}{d}$",
        fontsize=14,
        transform=ax.transAxes,
        rotation="vertical",
        color="r",
    )

    # Styling
    ax.set(
        title="Impact of $\lambda$ hyperparameter in loss function",
        xlabel="$\lambda$",
        xticks=x,
        xticklabels=lambda_values,
        ylabel="Test accuracy [%]",
    )
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.label.set_size(14)

    fig.tight_layout()
    fig.savefig("data/lambda_ablation_plot.png")


if __name__ == "__main__":
    main()

