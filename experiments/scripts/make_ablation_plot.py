import matplotlib.pyplot as plt
import seaborn as sns


def get_results():
    res = {
        # "augmentation": {
        #     "Without augmentation": (0.672482, 0.010338),
        #     "Only node feature masking": (0.759774, 0.004649),
        #     "Only edge dropping": (0.722319, 0.010389),
        #     "Both augmentation functions\n(Baseline)": (0.767847, 0.008448),
        # },
        # "encoder": {
        #     "MLP": (0.685274, 0.008973),
        #     "One layer GCN": (0.762237, 0.005997),
        #     "Two layer GCN\n(Baseline)": (0.771130, 0.007108),
        #     "Three layer GCN": (0.765213, 0.009903),
        # }
        "augmentation": {
            "No augmentation": (0.672482, 0.010338),
            "Node feature masking": (0.759774, 0.004649),
            "Edge dropping": (0.722319, 0.010389),
            "Both\n(Baseline)": (0.767847, 0.008448),
        },
        "encoder": {
            "MLP": (0.685274, 0.008973),
            "1-GCN": (0.762237, 0.005997),
            "2-GCN\n(Baseline)": (0.771130, 0.007108),
            "3-GCN": (0.765213, 0.009903),
        }
    }

    for s in res.keys():
        for k, v in res[s].items():
            res[s][k] = (v[0] * 100., v[1] * 100.)

    return res


def plot_ablation(results, ax):
    baseline_key = [k for k in results.keys() if "Baseline" in k][0]
    baseline_mean = results[baseline_key][0]

    scenarios_diff = [
        (key, v[0] - baseline_mean, v[1])
        for key, v in results.items()
    ]

    scenarios_diff = reversed(sorted(scenarios_diff, key=lambda x: x[1]))

    keys, mean_diffs, stds = zip(*scenarios_diff)

    x = range(len(keys))

    ax.plot(x, mean_diffs, linestyle="--", marker="o")
    ax.set(
        xlabel="",
        ylabel="Accuracy difference",
        xticks=x,
        xticklabels=keys,
    )
    ax.xaxis.set_tick_params(rotation=45, labelsize=12)
    ax.yaxis.label.set_size(14)


def main():
    results = get_results()

    fig, axs = plt.subplots(ncols=2, figsize=(15, 5))
    plt.rcParams.update({'axes.titlesize': 14})

    # Augmentations
    plot_ablation(results["augmentation"], axs[0])
    axs[0].set(title="Augmentation function impact")

    plot_ablation(results["encoder"], axs[1])
    axs[1].set(title="Encoder architecture impact")

    fig.tight_layout()
    fig.savefig("ablation_study.png")


if __name__ == "__main__":
    main()

