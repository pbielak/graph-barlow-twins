from copy import deepcopy
import os
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn.models import Node2Vec
from torch_geometric import nn as tgnn
from torch_geometric.nn.inits import uniform
from tqdm import tqdm
from tqdm import trange

from GCL import augmentors as A
from GCL.losses import BarlowTwins, BootstrapLatent, InfoNCE, JSD
from GCL.models import (
    BootstrapContrast,
    DualBranchContrast,
    SingleBranchContrast,
    WithinEmbedContrast,
)
from gssl.datasets import load_dataset

DATASETS = ["WikiCS", "Amazon-CS", "Amazon-Photo", "Coauthor-CS", "Coauthor-Physics"]
NUM_EPOCHS = 10
EMB_DIM = 256
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training_DeepWalk(data: Data):
    # Build model
    model = Node2Vec(
        edge_index=data.edge_index,
        embedding_dim=EMB_DIM,
        walk_length=30,
        context_size=5,
        walks_per_node=5,
        p=1,
        q=1,
        sparse=True,
    ).to(DEVICE)

    train_loader = model.loader(batch_size=1024, shuffle=True, num_workers=1)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=LR)

    # Train loop
    model.train()

    epoch_times = []

    for _ in trange(NUM_EPOCHS, desc="DeepWalk", leave=False):
        start = perf_counter()

        for pos_rw, neg_rw in tqdm(train_loader, desc="Batches", leave=False):
            optimizer.zero_grad()
            loss = model.loss(
                pos_rw=pos_rw.to(DEVICE),
                neg_rw=neg_rw.to(DEVICE),
            )
            loss.backward()
            optimizer.step()

        end = perf_counter()
        epoch_times.append(end - start)

    return epoch_times


class TwoLayerGCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self._conv1 = tgnn.GCNConv(in_dim, 2 * out_dim)
        self._conv2 = tgnn.GCNConv(2 * out_dim, out_dim)

        self._bn1 = nn.BatchNorm1d(2 * out_dim, momentum=0.01)  # same as `weight_decay = 0.99`

        self._act1 = nn.PReLU()

    def forward(self, x, edge_index, edge_weight=None):
        x = self._conv1(x, edge_index, edge_weight=edge_weight)
        x = self._bn1(x)
        x = self._act1(x)

        x = self._conv2(x, edge_index, edge_weight=edge_weight)

        return x


def run_training_DGI(data: Data):
    # Build model
    class _DGIEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self._encoder = TwoLayerGCN(
                in_dim=data.num_node_features,
                out_dim=EMB_DIM,
            )
            self._project = nn.Linear(
                in_features=EMB_DIM,
                out_features=EMB_DIM,
            )
            uniform(EMB_DIM, self._project.weight)

        @staticmethod
        def corruption(x, edge_index):
            return x[torch.randperm(x.size(0))], edge_index

        def forward(self, x, edge_index):
            z = self._encoder(x, edge_index)
            g = self._project(torch.sigmoid(z.mean(dim=0, keepdim=True)))
            zn = self._encoder(*self.corruption(x, edge_index))

            return z, g, zn

    model = _DGIEncoder().to(DEVICE)
    contrast_model = SingleBranchContrast(loss=JSD(), mode="G2L").to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train loop
    model.train()
    contrast_model.train()

    epoch_times = []

    for _ in trange(NUM_EPOCHS, desc="DGI", leave=False):
        start = perf_counter()

        optimizer.zero_grad()

        _z, _g, _zn = model(data.x.to(DEVICE), data.edge_index.to(DEVICE))
        loss = contrast_model(h=_z, g=_g, hn=_zn)

        loss.backward()
        optimizer.step()

        end = perf_counter()
        epoch_times.append(end - start)

    return epoch_times


def run_training_GMI():  # TODO: No implementation in GCL
    pass


def run_training_MVGRL(data: Data):
    # Build model
    class _MVGRLEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self._enc1 = TwoLayerGCN(
                in_dim=data.num_node_features,
                out_dim=EMB_DIM,
            )
            self._enc2 = TwoLayerGCN(
                in_dim=data.num_node_features,
                out_dim=EMB_DIM,
            )
            self._augmentor = (
                A.Identity(),
                A.PPRDiffusion(alpha=0.2),
            )
            self._project = torch.nn.Linear(
                in_features=EMB_DIM,
                out_features=EMB_DIM,
            )
            uniform(EMB_DIM, self._project.weight)

        @staticmethod
        def corruption(x, edge_index, edge_weight):
            return x[torch.randperm(x.size(0))], edge_index, edge_weight

        def forward(self, x, edge_index, edge_weight=None):
            aug1, aug2 = self._augmentor
            x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
            x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

            z1 = self._enc1(x1, edge_index1, edge_weight1)
            z2 = self._enc2(x2, edge_index2, edge_weight2)

            g1 = self._project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
            g2 = self._project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))

            z1n = self._enc1(*self.corruption(x1, edge_index1, edge_weight1))
            z2n = self._enc2(*self.corruption(x2, edge_index2, edge_weight2))

            return z1, z2, g1, g2, z1n, z2n

    model = _MVGRLEncoder().to(DEVICE)
    contrast_model = DualBranchContrast(loss=JSD(), mode="G2L").to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train loop
    model.train()
    contrast_model.train()

    epoch_times = []

    for _ in trange(NUM_EPOCHS, desc="MVGRL", leave=False):
        start = perf_counter()

        optimizer.zero_grad()

        _z1, _z2, _g1, _g2, _z1n, _z2n = model(
            data.x.to(DEVICE), data.edge_index.to(DEVICE)
        )
        loss = contrast_model(
            h1=_z1, h2=_z2,
            g1=_g1, g2=_g2,
            h1n=_z1n, h2n=_z2n,
        )

        loss.backward()
        optimizer.step()

        end = perf_counter()
        epoch_times.append(end - start)

    return epoch_times


def run_training_GRACE(data: Data, proj_dim: int = 256):
    # Build model
    class _GRACEEncoder(nn.Module):
        def __init__(self, augmentor):
            super().__init__()
            self._encoder = TwoLayerGCN(
                in_dim=data.num_node_features,
                out_dim=EMB_DIM,
            )
            self._augmentor = augmentor

            self.fc1 = nn.Linear(EMB_DIM, proj_dim)
            self.fc2 = nn.Linear(proj_dim, EMB_DIM)

        def forward(self, x, edge_index, edge_weight=None):
            aug1, aug2 = self._augmentor

            x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
            x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

            # z = self._encoder(x, edge_index, edge_weight)  # Not used during training
            z = None
            z1 = self._encoder(x1, edge_index1, edge_weight1)
            z2 = self._encoder(x2, edge_index2, edge_weight2)

            return z, z1, z2

        def project(self, z: torch.Tensor) -> torch.Tensor:
            z = F.elu(self.fc1(z))
            return self.fc2(z)

    _aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    _aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])

    model = _GRACEEncoder(augmentor=(_aug1, _aug2)).to(DEVICE)
    contrast_model = DualBranchContrast(
        loss=InfoNCE(tau=0.2),
        mode="L2L",
        intraview_negs=True,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train loop
    model.train()
    contrast_model.train()

    epoch_times = []

    for _ in trange(NUM_EPOCHS, desc="GRACE", leave=False):
        start = perf_counter()

        optimizer.zero_grad()

        _, _z1, _z2 = model(
            data.x.to(DEVICE), data.edge_index.to(DEVICE)
        )
        _h1, _h2 = [model.project(x) for x in [_z1, _z2]]
        loss = contrast_model(h1=_h1, h2=_h2)

        loss.backward()
        optimizer.step()

        end = perf_counter()
        epoch_times.append(end - start)

    return epoch_times


def run_training_BGRL(data: Data):
    # Build model
    class _TwoLayerGCNWithBN(nn.Module):
        def __init__(self):
            super().__init__()
            self._enc = TwoLayerGCN(
                in_dim=data.num_node_features,
                out_dim=EMB_DIM,
            )
            self._batch_norm = nn.BatchNorm1d(EMB_DIM, momentum=0.01)
            self._act = nn.PReLU()

        def forward(self, x, edge_index, edge_weight=None):
            x = self._enc(x, edge_index, edge_weight)
            x = self._batch_norm(x)
            x = self._act(x)
            return x

    class _BGRLEncoder(nn.Module):
        def __init__(self, augmentor):
            super().__init__()

            self._online_encoder = _TwoLayerGCNWithBN()
            self._target_encoder = None

            self._augmentor = augmentor

            self._predictor = torch.nn.Sequential(
                nn.Linear(EMB_DIM, 512),
                nn.BatchNorm1d(512, momentum=0.01),
                nn.PReLU(),
                nn.Linear(512, EMB_DIM),
                nn.BatchNorm1d(EMB_DIM, momentum=0.01),
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

        def forward(self, x, edge_index, edge_weight=None):
            aug1, aug2 = self._augmentor

            x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
            x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

            h1 = self._online_encoder(x1, edge_index1, edge_weight1)
            h2 = self._online_encoder(x2, edge_index2, edge_weight2)

            h1_pred = self._predictor(h1)
            h2_pred = self._predictor(h2)

            with torch.no_grad():
                h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
                h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)

            return h1, h2, h1_pred, h2_pred, h1_target, h2_target

    _aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
    _aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    model = _BGRLEncoder(augmentor=(_aug1, _aug2)).to(DEVICE)
    contrast_model = BootstrapContrast(
        loss=BootstrapLatent(),
        mode="L2L",
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train loop
    model.train()
    contrast_model.train()

    epoch_times = []

    for _ in trange(NUM_EPOCHS, desc="BGRL", leave=False):
        start = perf_counter()

        optimizer.zero_grad()

        _, _, _h1_pred, _h2_pred, _h1_target, _h2_target = model(
            data.x.to(DEVICE), data.edge_index.to(DEVICE),
        )
        loss = contrast_model(
            h1_pred=_h1_pred, h2_pred=_h2_pred,
            h1_target=_h1_target.detach(), h2_target=_h2_target.detach(),
        )

        loss.backward()
        optimizer.step()
        model.update_target_encoder(0.99)

        end = perf_counter()
        epoch_times.append(end - start)

    return epoch_times


def run_training_GBT(data: Data):
    # Build model
    class _GBTEncoder(nn.Module):
        def __init__(self, augmentor):
            super().__init__()
            self._enc = TwoLayerGCN(
                in_dim=data.num_node_features,
                out_dim=EMB_DIM,
            )
            self._augmentor = augmentor

        def forward(self, x, edge_index, edge_weight=None):
            aug1, aug2 = self._augmentor

            x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
            x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

            # z = self._enc(x, edge_index, edge_weight)  # Not used during training
            z = None

            z1 = self._enc(x1, edge_index1, edge_weight1)
            z2 = self._enc(x2, edge_index2, edge_weight2)

            return z, z1, z2

    _aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
    _aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    model = _GBTEncoder(augmentor=(_aug1, _aug2)).to(DEVICE)
    contrast_model = WithinEmbedContrast(loss=BarlowTwins()).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train loop
    model.train()
    contrast_model.train()

    epoch_times = []

    for _ in trange(NUM_EPOCHS, desc="GBT", leave=False):
        start = perf_counter()

        optimizer.zero_grad()

        _, z1, z2 = model(
            data.x.to(DEVICE), data.edge_index.to(DEVICE),
        )
        loss = contrast_model(h1=z1, h2=z2)

        loss.backward()
        optimizer.step()

        end = perf_counter()
        epoch_times.append(end - start)

    return epoch_times


def main():
    methods = {
        "DeepWalk": run_training_DeepWalk,
        "DGI": run_training_DGI,
        # "GMI": run_training_GMI,  # TODO: No implementation in GCL
        "MVGRL": run_training_MVGRL,  # TODO: yields OOM
        "GRACE": run_training_GRACE,
        "BGRL": run_training_BGRL,
        "GBT": run_training_GBT,
    }
    results = []

    for dataset in DATASETS:
        print(f"Dataset: {dataset}")
        for name, train_fn in methods.items():
            data, _ = load_dataset(name=dataset)

            try:
                epoch_times = train_fn(data=data)

                mean = np.mean(epoch_times)
                std = np.std(epoch_times, ddof=1)

                res = {
                    "dataset": dataset,
                    "method": name,
                    "epoch_times": epoch_times,
                    "mean": mean,
                    "std": std,
                    "status": "OK",
                }
                print(f"Method \"{name}\" took: {mean:.2f} +/ {std:.2f} [s] per epoch")
            except Exception as e:
                if "CUDA out of memory" in e.args[0]:
                    res = {
                        "dataset": dataset,
                        "method": name,
                        "epoch_times": [],
                        "mean": np.nan,
                        "std": np.nan,
                        "status": "OOM",
                    }
                    print(f"Method \"{name}\" failed with OOM: {e.args[0]}")
                else:
                    raise e


            results.append(res)

            torch.cuda.empty_cache()

    os.makedirs("data/time", exist_ok=True)

    df = pd.DataFrame.from_records(results)

    df.to_csv(path_or_buf="data/time/log.csv", index=False)

    df["value"] = df[["mean", "std"]].apply(
        lambda x: f"{x['mean']:.2f} +/- {x['std']:.2f} [s]",
        axis=1,
    )
    df.loc[df["status"] == "OOM", "value"] = "OOM"

    summary_df = (
        df.pivot(
            index="method",
            columns="dataset",
            values="value",
        )
        .reindex(list(methods.keys()))
        [DATASETS]
    )
    with open("data/time/summary_table.txt", "w") as fout:
        fout.write(summary_df.to_string())


if __name__ == "__main__":
    main()
