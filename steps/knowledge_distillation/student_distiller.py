from typing import Dict, Any, List, Literal, Tuple
import copy
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from zenml import step
from zenml.logger import get_logger
from utils.window_dataset import make_loader
from models.cnn_lstm import CNNLSTMWithAttention
from models.lstm_baseline import LSTM_Baseline

logger = get_logger(__name__)

StudentType   = Literal["cnn_lstm", "lstm"]
KDKind        = Literal["logits", "soft"]

def kd_loss_logit(student_out: torch.Tensor,
                  teacher_out: torch.Tensor,
                  hard_target: torch.Tensor,
                  hard_ratio: float = 0.3) -> torch.Tensor:
    """
    Logit-matching  + optional hard loss (MSE) on true labels.
    """
    mse_soft = F.mse_loss(student_out, teacher_out.detach())
    mse_hard = F.mse_loss(student_out, hard_target)
    return (1-hard_ratio) * mse_soft + hard_ratio * mse_hard


def kd_loss_soft(student_out: torch.Tensor,
                 teacher_out: torch.Tensor,
                 hard_target: torch.Tensor,
                 T: float = 3.0,
                 alpha: float = 0.7) -> torch.Tensor:
    """
    Soft-target KD (Hinton): KLDiv(student_T ‖ teacher_T) + hard loss.
    """
    # soften logits
    s_logp = F.log_softmax(student_out / T, dim=-1)
    t_prob = F.softmax(teacher_out.detach() / T, dim=-1)
    loss_soft = F.kl_div(s_logp, t_prob, reduction="batchmean") * T * T
    loss_hard = F.mse_loss(student_out, hard_target)
    return alpha * loss_soft + (1-alpha) * loss_hard

# Map for quick choice
KD_LOSSES = {"logits": kd_loss_logit, "soft": kd_loss_soft}

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Helper: build student model with reduced capacity
# ──────────────────────────────────────────────────────────────────────────────

def build_student(student_kind: StudentType,
                  n_features: int,
                  n_targets: int,
                  horizon: int,
                  teacher_hparams: Dict[str, Any]
                  ) -> nn.Module:
    """
    Produces a lighter student:
        * cnn_lstm – ½ filters, ½ hidden, 1 LSTM layer
        * lstm     – as in LSTM_Baseline (mean+max pooling)
    """
    if student_kind == "cnn_lstm":
        small_channels = [c // 2 for c in teacher_hparams["cnn_channels"]]
        model = CNNLSTMWithAttention(
            n_features=n_features,
            n_targets=n_targets,
            horizon=horizon,
            cnn_channels=small_channels,
            kernels=teacher_hparams["kernels"],
            lstm_hidden=int(teacher_hparams["hidden_lstm"] // 2),
            lstm_layers=1,
            dropout=float(teacher_hparams["dropout_rate"]),
        )
    elif student_kind == "lstm":
        model = LSTM_Baseline(
            n_features=n_features,
            n_targets=n_targets,
            horizon=horizon,
            cnn_channels=[], kernels=[],                   # ignored
            lstm_hidden=int(teacher_hparams["hidden_lstm"] // 2),
            lstm_layers=1,
            dropout=float(teacher_hparams["dropout_rate"]),
        )
    else:
        raise ValueError(f"Unknown student_kind: {student_kind}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 3.  ZenML step – distillation
# ──────────────────────────────────────────────────────────────────────────────

@step(enable_cache=False)
def student_distiller(                       # noqa: C901 (complexity – ok in step)
    train: Dict[str, pd.DataFrame],
    val: Dict[str, pd.DataFrame],
    seq_len: int,
    horizon: int,
    selected_columns: List[str],
    teacher: nn.Module,                   # Teacher already trained
    teacher_hparams: Dict[str, Any],
    student_kind: StudentType = "lstm",
    kd_kind: KDKind = "soft",
    epochs: int = 30,
    early_stop_epochs: int = 5,
    batch: int = 32,
    lr: float = 1e-3,
) -> nn.Module:
    """
    Distils `teacher` into a smaller `student_kind` model using `kd_kind` loss.

    Returns
    -------
    student : nn.Module
        Trained/distilled student model.
    """

    # ── 0. Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device).eval()                         # freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False

    # ── 1. Data loaders (reuse existing utility) ───────────────────────────
    train_loader, _ = make_loader(
        dfs=train,
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch,
        shuffle=True,
        target_cols=selected_columns,
    )
    val_loader, _ = make_loader(
        dfs=val,
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch,
        shuffle=False,
        target_cols=selected_columns,
    )

    # sample shapes
    sample_X, sample_y = next(iter(train_loader))
    _, _, n_features = sample_X.shape
    _, _, n_targets  = sample_y.shape
    logger.info(f"Distillation | X {sample_X.shape}, y {sample_y.shape}")

    # ── 2. Build student ───────────────────────────────────────────────────
    student = build_student(student_kind,
                            n_features, n_targets, horizon,
                            teacher_hparams).to(device)
    optimizer = Adam(student.parameters(), lr=lr)
    kd_fn     = KD_LOSSES[kd_kind]

    # ── 3. Training loop w/ early stopping ────────────────────────────────
    best_val, patience = float("inf"), 0
    best_state: Tuple[int, dict] | None = None

    for epoch in range(epochs):
        # —— train ——
        student.train()
        run_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            # forward teacher & student
            with torch.no_grad():
                t_out = teacher(X)                # (B, H, T)
            s_out = student(X)                    # (B, H, T)

            loss = kd_fn(s_out, t_out, y)         # KD loss
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(X)

        train_loss = run_loss / len(train_loader.dataset)

        # —— val ——
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                v_loss = kd_fn(student(X), teacher(X), y)
                val_loss += v_loss.item() * len(X)
        val_loss /= len(val_loader.dataset)

        logger.info(f"[{epoch:02d}] distill "
                    f"train={train_loss:.4f}  val={val_loss:.4f}")

        # early-stopping
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(student.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_epochs:
                logger.info("Early-stopping.")
                break

    # ── 4. Restore best weights ───────────────────────────────────────────
    if best_state:
        student.load_state_dict(best_state)
    else:
        logger.warning("No improvement – final weights kept.")

    logger.info(f"✅ Student ({student_kind}, {kd_kind}) best val {best_val:.4f}")
    return student
