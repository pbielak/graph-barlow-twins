import json
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from tqdm.auto import tqdm

from gssl.augment import GraphAugmentor
from gssl.datasets import load_dataset
from gssl.full_batch.encoders import TwoLayerGCNEncoder, ThreeLayerGCNEncoder
from gssl.full_batch.model import FullBatchModel
from gssl.tasks import evaluate_node_classification_acc
from gssl.utils import seed


class OneLayerGCNEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.conv = GCNConv(in_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)

        return x


class MLPEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim, momentum=0.01),
            nn.PReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim, momentum=0.01),
            nn.PReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, edge_index):
        return self.layers(x)


def evaluate_single_graph_full_batch_model(
    dataset_name: str,
    params: dict,
    aug_params: dict,
) -> dict:
    data, masks = load_dataset(name=dataset_name)

    test_accuracies = []

    #for i in tqdm(range(20), desc="Splits"):
    for i in tqdm(range(5), desc="Splits"):
        augmentor = GraphAugmentor(
            p_x_1=aug_params["p_x_1"],
            p_e_1=aug_params["p_e_1"],
        )

        encoder = params["encoder_cls"](
            in_dim=data.num_node_features,
            out_dim=params["emb_dim"],
        )

        model = FullBatchModel(
            encoder=encoder,
            augmentor=augmentor,
            lr_base=params["lr_base"],
            total_epochs=params["total_epochs"],
            warmup_epochs=params["warmup_epochs"],
            use_pytorch_eval_model=params["use_pytorch_eval_model"],
        )

        model.fit(data=data)

        z = model.predict(data=data)
        accuracy = evaluate_node_classification_acc(
            z=z, data=data, masks=masks[i],
            use_pytorch=params["use_pytorch_eval_model"],
        )

        test_accuracies.append(accuracy["test"])

    statistics = {
        "mean": np.mean(test_accuracies),
        "std": np.std(test_accuracies, ddof=1),
    }
    return statistics


def main():
    seed()

    dataset_name = "WikiCS"

    default_params = dict(
        total_epochs=1000,
        warmup_epochs=100,
        use_pytorch_eval_model=False,
        emb_dim=256,
        lr_base=5e-4,
    )
    default_encoder_cls = TwoLayerGCNEncoder

    default_aug_params = dict(p_x_1=0.1, p_e_1=0.2)

    # -- Scenarios --
    scenario_params = {
        # Augmentation functions
        "no_augmentation": {
            "params": {"encoder_cls": default_encoder_cls, **default_params},
            "aug_params": dict(p_x_1=0, p_e_1=0),
        },
        "only_node_feature_masking": {
            "params": {"encoder_cls": default_encoder_cls, **default_params},
            "aug_params": dict(p_x_1=default_aug_params["p_x_1"], p_e_1=0),
        },
        "only_edge_dropping": {
            "params": {"encoder_cls": default_encoder_cls, **default_params},
            "aug_params": dict(p_x_1=0, p_e_1=default_aug_params["p_e_1"]),
        },
        "both_augmentations": {
            "params": {"encoder_cls": default_encoder_cls, **default_params},
            "aug_params": default_aug_params,
        },
        # Encoders
        "mlp_encoder": {
            "params": {"encoder_cls": MLPEncoder, **default_params},
            "aug_params": default_aug_params,
        },
        "one_layer_gcn": {
            "params": {"encoder_cls": OneLayerGCNEncoder, **default_params},
            "aug_params": default_aug_params,
        },
        "two_layer_gcn": {
            "params": {"encoder_cls": TwoLayerGCNEncoder, **default_params},
            "aug_params": default_aug_params,
        },
        "three_layer_gcn": {
            "params": {"encoder_cls": ThreeLayerGCNEncoder, **default_params},
            "aug_params": default_aug_params,
        },
    }

    records = []

    for name, sparams in tqdm(
        iterable=scenario_params.items(),
        desc="Ablation study scenarios",
    ):
        statistics = evaluate_single_graph_full_batch_model(
            dataset_name=dataset_name,
            **sparams,
        )
        records.append({"name": name, **statistics})

    df = pd.DataFrame.from_records(records)
    print(df)


if __name__ == "__main__":
    main()
