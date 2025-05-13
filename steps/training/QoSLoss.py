import torch
import torch.nn as nn

class QoSLoss(nn.Module):
    def __init__(self, over_weight: float = 1.0, under_weight: float = 10.0):
        super().__init__()
        self.over_w = over_weight
        self.under_w = under_weight

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff  = preds - target
        over  = torch.clamp(diff,     min=0.0)
        under = torch.clamp(-diff,    min=0.0)
        # weighted sum normalized by element count
        loss = (self.over_w * over.sum() + self.under_w * under.sum()) / diff.numel()
        return loss
