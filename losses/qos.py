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
