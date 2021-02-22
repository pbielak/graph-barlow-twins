import torch

EPS = 1e-15


def barlow_twins_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
) -> torch.Tensor:
    batch_size = z_a.size(0)
    feature_dim = z_a.size(1)
    _lambda = 1 / feature_dim

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
