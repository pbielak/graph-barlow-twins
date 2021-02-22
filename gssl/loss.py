import torch

EPS = 1e-15


def _cross_correlation_matrix(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
) -> torch.Tensor:
    batch_size = z_a.size(0)

    # Apply batch normalization
    z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + EPS)
    z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + EPS)

    # Cross-correlation matrix
    c = (z_a_norm.T @ z_b_norm) / batch_size

    return c


def barlow_twins_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
) -> torch.Tensor:
    feature_dim = z_a.size(1)
    _lambda = 1 / feature_dim

    # Cross-correlation matrix
    c = _cross_correlation_matrix(z_a=z_a, z_b=z_b)

    # Loss function
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (
        (1 - c.diagonal()).pow(2).sum()
        + _lambda * c[off_diagonal_mask].pow(2).sum()
    )

    return loss


def hilbert_schmidt_independence_criterion(
    z_a: torch.Tensor,
    z_b: torch.Tensor
) -> torch.Tensor:
    feature_dim = z_a.size(1)
    _lambda = 1 / feature_dim

    # Cross-correlation matrix
    c = _cross_correlation_matrix(z_a=z_a, z_b=z_b)

    # Loss function
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (
        (1 - c.diagonal()).pow(2).sum()
        + _lambda * (1 + c[off_diagonal_mask]).pow(2).sum()
    )

    return loss


def get_loss(loss_name: str):
    if loss_name == "barlow_twins":
        return barlow_twins_loss
    elif loss_name == "hsic":
        return hilbert_schmidt_independence_criterion
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

