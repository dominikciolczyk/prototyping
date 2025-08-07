from cProfile import label

import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import set_plot_style
from qos import AsymmetricL1, AsymmetricSmoothL1

set_plot_style()

alpha = 10
beta = 3.0

loss_l1_fn = AsymmetricL1(alpha=alpha)
loss_smooth_fn = AsymmetricSmoothL1(beta=beta, alpha=alpha)

errors_custom = torch.linspace(-5, 5, steps=500)
y_true = torch.tensor([0.0])

loss_l1 = torch.stack([loss_l1_fn(e.view(1), y_true) for e in errors_custom])
loss_smooth = torch.stack([loss_smooth_fn(e.view(1), y_true) for e in errors_custom])


plt.figure(figsize=(7, 4))
plt.plot(errors_custom.numpy(), loss_l1.numpy(), label="Asymmetric MAE", linestyle="--", color="C0")
plt.plot(errors_custom.numpy(), loss_smooth.numpy(), label="Asymmetric Smooth MAE", linestyle="-", color="C1")

plt.axvline(x=0, color="gray", linestyle=":", linewidth=1)
plt.axvline(x=beta, color="gray", linestyle=":", linewidth=0.8)
plt.axvline(x=-beta, color="gray", linestyle=":", linewidth=0.8)

plt.xlabel("błąd predykcji $e_n = \\hat{y}_n - y_n$")
plt.ylabel("wartość funkcji straty")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("asymmetric_loss_comparison.pdf", format="pdf")
plt.close()
