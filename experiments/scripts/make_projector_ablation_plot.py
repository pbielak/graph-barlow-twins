import matplotlib.pyplot as plt
import pandas as pd


def make_size_vs_acc_plot(data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.rcParams.update({'axes.titlesize': 14})

    projector_dims = data["name"]
    mean_acc = data["acc_mean"] * 100.
    std_acc = data["acc_std"] * 100.

    x = range(len(projector_dims))

    # Plot accuracy results
    ax.plot(x, mean_acc, linestyle="--", marker="o")
    ax.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, alpha=0.3)

    # Mark results for "no projector"
    default_lambda_index = list(projector_dims).index(-1)  # NOTE: hard-coded
    ax.axvline(
        x=default_lambda_index,
        linestyle="-.",
        color="r",
    )
    ax.text(
        x=0.03,
        y=0.40,
        s="No projector",
        fontsize=14,
        transform=ax.transAxes,
        rotation="vertical",
        color="r",
    )

    # Styling
    ax.set(
        title="Impact of projector dimensionality",
        xlabel="Projector output dimensionality",
        xticks=x,
        xticklabels=projector_dims,
        ylabel="Test accuracy [%]",
        ylim=(70, 80),
    )
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.label.set_size(14)

    fig.tight_layout()
    fig.savefig("data/projector_ablation_plot_size_vs_acc.png")


def make_time_vs_acc_plot(data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.rcParams.update({'axes.titlesize': 14})

    projector_dims = data["name"]
    mean_acc = data["acc_mean"] * 100.
    std_acc = data["acc_std"] * 100.

    mean_time = data["time_mean"]
    std_time = data["time_std"]

    ax.scatter(x=mean_time, y=mean_acc)

    for time, acc, dim in zip(mean_time, mean_acc, projector_dims):
        if dim == -1:
            dim = "No projector"
            color = "r"
        else:
            color = "b"
        ax.text(x=time * 1.01, y=acc, s=str(dim), color=color)

    row = data[data["name"] == -1]
    ax.scatter(x=row["time_mean"], y=row["acc_mean"] * 100.0, color="r")

    # Styling
    ax.set(
        title="Impact of projector dimensionality on accuracy and time",
        xlabel="Training time [s]",
        ylabel="Test accuracy [%]",
        xscale="log",
        ylim=(70, 80),
    )
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.label.set_size(14)

    fig.tight_layout()
    fig.savefig("data/projector_ablation_plot_time_vs_acc.png")


def main():
    data = pd.read_csv("data/projector_ablation_results.csv")
    data = data.sort_values(by="name")

    make_size_vs_acc_plot(data=data)
    make_time_vs_acc_plot(data=data)


if __name__ == "__main__":
    main()

