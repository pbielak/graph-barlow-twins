from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from tqdm.auto import tqdm

from gssl.augment import GraphAugmentor
from gssl.datasets import load_dataset
from gssl.full_batch.encoders import TwoLayerGCNEncoder
from gssl.tasks import evaluate_node_classification_acc
from gssl.utils import plot_vectors, seed


# Copy loss implementation and add "lambda" as function argument
EPS = 1e-15


def barlow_twins_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    _lambda: float,
) -> torch.Tensor:
    batch_size = z_a.size(0)
    feature_dim = z_a.size(1)

    # Apply batch normalization
    z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + EPS)
    z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + EPS)

    # Cross-correlation matrix
    c = (z_a_norm.T @ z_b_norm) / batch_size

    # Loss function
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (
        (1 - c.diagonal()).pow(2).sum()
        + _lambda * c[off_diagonal_mask].pow(2).sum()
    )

    return loss


# Copy "full-batch" model implementation

class FullBatchModel:

    def __init__(
        self,
        encoder: nn.Module,
        augmentor: GraphAugmentor,
        lr_base: float,
        total_epochs: int,
        warmup_epochs: int,
        use_pytorch_eval_model: bool,
        _lambda: float,
    ):
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._encoder = encoder.to(self._device)

        self._optimizer = torch.optim.AdamW(
            params=self._encoder.parameters(),
            lr=lr_base,
            weight_decay=1e-5,
        )
        self._scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=self._optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=total_epochs,
        )

        self._augmentor = augmentor

        self._total_epochs = total_epochs

        self._use_pytorch_eval_model = use_pytorch_eval_model

        self._lambda = _lambda

    def fit(
        self,
        data: Data,
        logger: Optional[SummaryWriter] = None,
        log_interval: Optional[int] = None,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> dict:
        self._encoder.train()
        logs = {
            "log_epoch": [],
            "train_accuracies": [],
            "val_accuracies": [],
            "test_accuracies": [],
            "z": [],
        }

        data = data.to(self._device)

        for epoch in tqdm(iterable=range(self._total_epochs), leave=False):
            self._optimizer.zero_grad()

            (x_a, ei_a), (x_b, ei_b) = self._augmentor(data=data)

            z_a = self._encoder(x=x_a, edge_index=ei_a)
            z_b = self._encoder(x=x_b, edge_index=ei_b)

            loss = barlow_twins_loss(z_a=z_a, z_b=z_b, _lambda=self._lambda)

            loss.backward()

            # Save loss on every epoch
            if logger is not None:
                logger.add_scalar("Loss", loss.item(), epoch)

            # Log other metrics only in given interval
            if log_interval is not None and epoch % log_interval == 0:
                assert logger is not None

                z = self.predict(data=data)
                self._encoder.train()  # Predict sets `eval()` mode

                logger.add_figure(
                    "latent",
                    plot_vectors(z, labels=data.y.cpu()),
                    epoch
                )

                accs = evaluate_node_classification_acc(
                    z, data, masks=masks,
                    use_pytorch=self._use_pytorch_eval_model,
                )

                logger.add_scalar("acc/train", accs["train"], epoch)
                logger.add_scalar("acc/val", accs["val"], epoch)
                logger.add_scalar("acc/test", accs["test"], epoch)

                logs["log_epoch"].append(epoch)
                logs["train_accuracies"].append(accs["train"])
                logs["val_accuracies"].append(accs["val"])
                logs["test_accuracies"].append(accs["test"])
                logs["z"].append(deepcopy(z))

                logger.add_scalar("norm", z.norm(dim=1).mean(), epoch)

            self._optimizer.step()
            self._scheduler.step()

        # Save all metrics at the end
        if logger is not None:
            z = self.predict(data=data)
            self._encoder.train()  # Predict sets `eval()` mode

            accs = evaluate_node_classification_acc(
                z, data, masks=masks,
                use_pytorch=self._use_pytorch_eval_model,
            )

            logger.add_figure(
                "latent",
                plot_vectors(z, labels=data.y.cpu()),
                self._total_epochs
            )
            logger.add_scalar("acc/train", accs["train"], self._total_epochs)
            logger.add_scalar("acc/val", accs["val"], self._total_epochs)
            logger.add_scalar("acc/test", accs["test"], self._total_epochs)

            logger.add_scalar("norm", z.norm(dim=1).mean(), self._total_epochs)

            logs["log_epoch"].append(self._total_epochs)
            logs["train_accuracies"].append(accs["train"])
            logs["val_accuracies"].append(accs["val"])
            logs["test_accuracies"].append(accs["test"])
            logs["z"].append(deepcopy(z))

        data = data.to("cpu")

        return logs

    def predict(self, data: Data) -> torch.Tensor:
        self._encoder.eval()

        with torch.no_grad():
            z = self._encoder(
                x=data.x.to(self._device),
                edge_index=data.edge_index.to(self._device),
            )

            return z.cpu()


def evaluate_single_graph_full_batch_model(
    dataset_name: str,
    params: dict,
    aug_params: dict,
) -> dict:
    data, masks = load_dataset(name=dataset_name)

    test_accuracies = []

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
            _lambda=params["_lambda"],
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
        encoder_cls=TwoLayerGCNEncoder,
        total_epochs=1000,
        warmup_epochs=100,
        use_pytorch_eval_model=False,
        emb_dim=256,
        lr_base=5e-4,
    )

    default_aug_params = dict(p_x_1=0.1, p_e_1=0.2)

    _lambda_values = [
        2,
        1,
        0.5,
        0.1,
        0.01,
        0.001,
        0,
        1 / default_params["emb_dim"],
    ]

    records = []

    for _lambda in tqdm(
        iterable=_lambda_values,
        desc="Lambda values",
    ):
        statistics = evaluate_single_graph_full_batch_model(
            dataset_name=dataset_name,
            params={**default_params, "_lambda": _lambda},
            aug_params=default_aug_params,
        )
        records.append({"name": _lambda, **statistics})

        (
            pd.DataFrame
            .from_records(records)
            .to_csv(
                path_or_buf="data/lambda_ablation_results.csv",
                index=False,
            )
        )


if __name__ == "__main__":
    main()
