from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric import nn as tgnn
from tqdm.auto import tqdm

from gssl.loss import get_loss
from gssl.tasks import evaluate_node_classification
from gssl.utils import plot_vectors


class Model:

    def __init__(
        self,
        feature_dim: int,
        emb_dim: int,
        loss_name: str,
        p_x: float,
        p_e: float,
        lr_base: float,
        total_epochs: int,
        warmup_epochs: int,
    ):
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._encoder = GCNEncoder(
            in_dim=feature_dim, out_dim=emb_dim
        ).to(self._device)

        self._loss_fn = get_loss(loss_name=loss_name)

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

        self._p_x = p_x
        self._p_e = p_e

        self._total_epochs = total_epochs

        self._use_pytorch_eval_model = False

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

        for epoch in tqdm(iterable=range(self._total_epochs)):
            self._optimizer.zero_grad()

            (x_a, ei_a), (x_b, ei_b) = augment(
                data=data, p_x=self._p_x, p_e=self._p_e,
            )

            z_a = self._encoder(x=x_a, edge_index=ei_a)
            z_b = self._encoder(x=x_b, edge_index=ei_b)

            loss = self._loss_fn(z_a=z_a, z_b=z_b)

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

                accs = evaluate_node_classification(
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

            accs = evaluate_node_classification(
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


class GCNEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self._conv1 = tgnn.GCNConv(in_dim, 2 * out_dim)
        self._conv2 = tgnn.GCNConv(2 * out_dim, out_dim)

        self._bn1 = nn.BatchNorm1d(2 * out_dim, momentum=0.01)  # same as `weight_decay = 0.99`

        self._act1 = nn.PReLU()

    def forward(self, x, edge_index):
        x = self._conv1(x, edge_index)
        x = self._bn1(x)
        x = self._act1(x)

        x = self._conv2(x, edge_index)

        return x


def augment(data: Data, p_x: float, p_e: float):
    device = data.x.device

    x = data.x
    num_fts = x.size(-1)

    ei = data.edge_index
    num_edges = ei.size(-1)

    x_a = bernoulli_mask(size=(1, num_fts), prob=p_x).to(device) * x
    x_b = bernoulli_mask(size=(1, num_fts), prob=p_x).to(device) * x

    ei_a = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.]
    ei_b = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.]

    return (x_a, ei_a), (x_b, ei_b)


def bernoulli_mask(size: Union[int, Tuple[int, ...]], prob: float):
    return torch.bernoulli((1 - prob) * torch.ones(size))
