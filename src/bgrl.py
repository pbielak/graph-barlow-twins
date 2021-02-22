from copy import deepcopy
from typing import Dict, List, Union

import numpy as np
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.data import NeighborSampler
from torch_geometric.data.sampler import EdgeIndex
from torch_geometric import nn as tgnn
from tqdm import tqdm

from gssl.augment import GraphAugmentor
from gssl.batched.encoders import get_inference_loader
from gssl.tasks import evaluate_node_classification_acc


class TwoLayerGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()

        self._conv1 = tgnn.GCNConv(in_dim, hidden_dim)
        self._conv2 = tgnn.GCNConv(hidden_dim, out_dim)

        self._bn1 = nn.BatchNorm1d(hidden_dim, momentum=0.01)  # same as `weight_decay = 0.99`
        self._bn2 = nn.BatchNorm1d(out_dim, momentum=0.01)

        self._act1 = nn.PReLU()
        self._act2 = nn.PReLU()

    def forward(self, x, edge_index):
        x = self._conv1(x, edge_index)
        x = self._bn1(x)
        x = self._act1(x)

        x = self._conv2(x, edge_index)
        x = self._bn2(x)
        x = self._act2(x)

        return x


class BGRL(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        augmentor: GraphAugmentor,
        out_dim: int,
        pred_dim: int,
    ):
        super().__init__()

        self._online_encoder = encoder
        self._target_encoder = None

        self._augmentor = augmentor

        self._predictor = torch.nn.Sequential(
            nn.Linear(out_dim, pred_dim),
            nn.BatchNorm1d(pred_dim, momentum=0.01),
            nn.PReLU(),
            nn.Linear(pred_dim, out_dim),
            nn.BatchNorm1d(out_dim, momentum=0.01),
            nn.PReLU(),
        )

    def get_target_encoder(self):
        if self._target_encoder is None:
            self._target_encoder = deepcopy(self._online_encoder)

            for p in self._target_encoder.parameters():
                p.requires_grad = False

        return self._target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(
            self.get_target_encoder().parameters(),
            self._online_encoder.parameters(),
        ):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, data: Data):
        (x1, edge_index1), (x2, edge_index2) = self._augmentor(data=data)

        h1 = self._online_encoder(x1, edge_index1)
        h2 = self._online_encoder(x2, edge_index2)

        h1_pred = self._predictor(h1)
        h2_pred = self._predictor(h2)

        with torch.no_grad():
            h1_target = self.get_target_encoder()(x1, edge_index1)
            h2_target = self.get_target_encoder()(x2, edge_index2)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target

    def predict(self, data: Data):
        with torch.no_grad():
            return self._online_encoder(data.x, data.edge_index).cpu()


class BatchedBGRL(BGRL):
    def __init__(
        self,
        encoder: nn.Module,
        augmentor: GraphAugmentor,
        out_dim: int,
        pred_dim: int,
        inference_batch_size: int,
    ):
        super().__init__(
            encoder=encoder,
            augmentor=augmentor,
            out_dim=out_dim,
            pred_dim=pred_dim,
        )
        self._inference_batch_size = inference_batch_size

    def forward(self, x: torch.Tensor, adjs: List[EdgeIndex]):
        (x1, edge_indexes1), (x2, edge_indexes2) = self._augmentor.augment_batch(
            x=x, adjs=adjs
        )
        sizes = [adj.size[1] for adj in adjs]

        h1 = self._online_encoder(x=x1, edge_indexes=edge_indexes1, sizes=sizes)
        h2 = self._online_encoder(x=x2, edge_indexes=edge_indexes2, sizes=sizes)

        h1_pred = self._predictor(h1)
        h2_pred = self._predictor(h2)

        with torch.no_grad():
            h1_target = self.get_target_encoder()(x=x1, edge_indexes=edge_indexes1, sizes=sizes)
            h2_target = self.get_target_encoder()(x=x2, edge_indexes=edge_indexes2, sizes=sizes)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target

    def predict(self, data: Data):
        with torch.no_grad():
            h = self._online_encoder.inference(
                x_all=data.x,
                edge_index_all=data.edge_index,
                inference_batch_size=self._inference_batch_size,
                device=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            ).cpu()

        return h


class BatchedGAT4BGRL(nn.Module):
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 256,
        heads: int = 4,
    ):
        super().__init__()


        self.convs = nn.ModuleList([
            tgnn.GATConv(in_dim, hidden, heads=heads, concat=True),
            tgnn.GATConv(heads * hidden, hidden, heads=heads, concat=True),
            tgnn.GATConv(heads * hidden, out_dim, heads=heads, concat=False),
        ])
        self.skips = nn.ModuleList([
            nn.Linear(in_dim, heads * hidden),
            nn.Linear(heads * hidden, heads * hidden),
            nn.Linear(heads * hidden, out_dim),
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(heads * hidden, momentum=0.01),
            nn.BatchNorm1d(heads * hidden, momentum=0.01),
            nn.BatchNorm1d(out_dim, momentum=0.01),
        ])


    def forward(self, x, edge_indexes, sizes):
        for i, (edge_index, size) in enumerate(zip(edge_indexes, sizes)):
            x_target = x[:size]

            x = self.convs[i]((x, x_target), edge_index)
            x = x + self.skips[i](x_target)

            x = self.bns[i](x)
            x = F.elu(x)

        return x

    def inference(self, x_all, edge_index_all, inference_batch_size, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers, leave=False)
        pbar.set_description("Evaluation")

        inference_loader = get_inference_loader(
            edge_index=edge_index_all,
            batch_size=inference_batch_size,
            num_nodes=x_all.shape[0],
        )

        for i in range(len(self.convs)):
            pbar.set_description(f"Evaluation [{i+1}/{len(self.convs)}]")
            xs = []
            for batch_size, n_id, adj in inference_loader:
                edge_index, _, size = adj.to(device)

                x = x_all[n_id].to(device)
                x_target = x[:size[1]]

                x = self.convs[i]((x, x_target), edge_index)
                x = x + self.skips[i](x_target)

                x = self.bns[i](x)
                x = F.elu(x)

                xs.append(x.cpu())
                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

    @property
    def num_layers(self):
        return len(self.convs)


def compute_tau(
    epoch: int,
    total_epochs: int,
    tau_base: float = 0.99,
) -> float:
    return (
        1.0 - (
            ((1.0 - tau_base) / 2.0)
            * (np.cos((epoch * np.pi) / total_epochs) + 1.0)
        )
    )


def test(
    model: Union[BGRL, BatchedBGRL],
    data: Data,
    masks: Dict[str, torch.Tensor],
    use_pytorch_eval_model: bool,
    device: torch.device,
):
    model.eval()
    z = model.predict(data=data.to(device))

    accs = evaluate_node_classification_acc(
        z=z, data=data, masks=masks, use_pytorch=use_pytorch_eval_model,
    )

    return z, accs


def train_batched(
    model: BatchedBGRL,
    contrast_model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    scheduler: LinearWarmupCosineAnnealingLR,
    data: Data,
    loader: NeighborSampler,
    device: torch.device,
    tau: float,
):
    model.train()
    contrast_model.train()

    total_loss = 0

    for _, n_id, adjs in tqdm(iterable=loader, desc="Batches", leave=False):
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()

        _, _, h1_pred, h2_pred, h1_target, h2_target = model(
            x=data.x[n_id].to(device), adjs=adjs,
        )
        loss = contrast_model(
            h1_pred=h1_pred, h2_pred=h2_pred,
            h1_target=h1_target.detach(), h2_target=h2_target.detach(),
        )

        loss.backward()
        total_loss += loss.item()

        optimizer.step()
        model.update_target_encoder(tau)

    scheduler.step()

    avg_loss = total_loss / len(loader)
    return avg_loss
