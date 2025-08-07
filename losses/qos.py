"""
QoS-aware losses ― under-provisioning is penalised more than over-provisioning.
y_true, y_pred are in *absolute* utilisation units (e.g. vCPU, GiB RAM,…).
"""

import torch
from torch import nn

class AsymmetricL1(nn.Module):
    """|err| if over-provision,   α*|err| if under-provision (α>1)."""
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        loss = torch.where(
            diff < 0,                                # under-provision
            self.alpha * diff.abs(),
            diff.abs(),
        )
        return loss.mean()


class AsymmetricSmoothL1(nn.Module):
    """Huber-like but asymmetric."""
    def __init__(self, beta: float, alpha: float):
        super().__init__()
        self.beta, self.alpha = beta, alpha

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        abs_diff = diff.abs()
        factor = torch.where(diff < 0, self.alpha, 1.0)
        loss = torch.where(
            abs_diff < self.beta,
            0.5 * factor * abs_diff ** 2 / self.beta,
            factor * (abs_diff - 0.5 * self.beta),
        )
        return loss.mean()


class AsymmetricCLoss(nn.Module):
    """
    Asymmetric correntropy-inspired loss (AC-loss).
    Introduces exponential decay with asymmetry controlled by tau ∈ (0,1).
    """
    def __init__(self, tau: float, sigma: float):
        super().__init__()
        self.tau = tau
        self.sigma = sigma

    def forward(self, y_pred, y_true):
        diff = y_true - y_pred
        tau = torch.tensor(self.tau, dtype=diff.dtype, device=diff.device)
        sigma_sq = self.sigma ** 2

        loss = torch.where(
            diff >= 0,
            1 - torch.exp(-tau * diff ** 2 / (2 * sigma_sq)),
            1 - torch.exp(-(1 - tau) * diff ** 2 / (2 * sigma_sq)),
        )
        return loss.mean()


class AssymetricHuberLoss(nn.Module):
    def __init__(self, alfa: float = 10.0, beta: float = 3.0):
        super().__init__()
        self.alfa = alfa
        self.delta = beta
        self.base_huber = nn.HuberLoss(delta=self.delta, reduction='none')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = self.base_huber(y_pred, y_true)

        mask = (y_pred < y_true).float()
        weight = 1.0 + mask * (self.alfa - 1.0)

        return (loss * weight).mean()
