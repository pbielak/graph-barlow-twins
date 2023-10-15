from copy import deepcopy
from time import perf_counter
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
from gssl.loss import barlow_twins_loss
from gssl.tasks import evaluate_node_classification_acc
from gssl.utils import plot_vectors, seed


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
        projector_dim: int,
    ):
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._encoder = encoder.to(self._device)

        if projector_dim == -1:
            projector = nn.Identity()
        else:
            projector = nn.Linear(
                self._encoder._conv2.out_channels,  # NOTE: hard-coded
                projector_dim,
            )
        self._projector = projector.to(self._device)

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

    def fit(
        self,
        data: Data,
        logger: Optional[SummaryWriter] = None,
        log_interval: Optional[int] = None,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> dict:
        self._encoder.train()
        self._projector.train()
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

            h_a = self._encoder(x=x_a, edge_index=ei_a)
            h_b = self._encoder(x=x_b, edge_index=ei_b)

            z_a = self._projector(h_a)
            z_b = self._projector(h_b)

            loss = barlow_twins_loss(z_a=z_a, z_b=z_b)

            loss.backward()

            # Save loss on every epoch
            if logger is not None:
                logger.add_scalar("Loss", loss.item(), epoch)

            # Log other metrics only in given interval
            if log_interval is not None and epoch % log_interval == 0:
                assert logger is not None

                z = self.predict(data=data)
                self._encoder.train()  # Predict sets `eval()` mode
                self._projector.train()

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
            self._projector.train()

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
        self._projector.eval()

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
    times = []

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
            projector_dim=params["projector_dim"],
        )

        st = perf_counter()
        model.fit(data=data)
        end = perf_counter()

        z = model.predict(data=data)
        accuracy = evaluate_node_classification_acc(
            z=z, data=data, masks=masks[i],
            use_pytorch=params["use_pytorch_eval_model"],
        )

        test_accuracies.append(accuracy["test"])
        times.append(end - st)

    statistics = {
        "acc_mean": np.mean(test_accuracies),
        "acc_std": np.std(test_accuracies, ddof=1),
        "time_mean": np.mean(times),
        "time_std": np.std(times, ddof=1),
        "all_acc": test_accuracies,
        "all_times": times,
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

    projector_dims = [
        -1,  # Do not use projector
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
    ]

    records = []

    for projector_dim in tqdm(
        iterable=projector_dims,
        desc="Projector dims",
    ):
        statistics = evaluate_single_graph_full_batch_model(
            dataset_name=dataset_name,
            params={**default_params, "projector_dim": projector_dim},
            aug_params=default_aug_params,
        )
        records.append({"name": projector_dim, **statistics})

        (
            pd.DataFrame
            .from_records(records)
            .to_pickle(path="data/projector_ablation_results.pkl")
        )


if __name__ == "__main__":
    main()
