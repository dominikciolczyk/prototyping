from typing import Dict, Any, List, Literal, Tuple
import copy
import pandas as pd
from torch import nn
from torch.optim import Adam
from zenml import step
from utils.window_dataset import make_loader
from models.cnn_lstm import CNNLSTMWithAttention
from models.lstm_baseline import LSTM_Baseline
import torch
import torch.nn.functional as F
from zenml.logger import get_logger

logger = get_logger(__name__)

StudentType   = Literal["cnn_lstm", "lstm"]
KDKind        = Literal["logits", "soft"]

def kd_loss_regression(
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        hard_target: torch.Tensor,
        kd_ratio: float,
        T: float,
        use_temperature: bool,
) -> torch.Tensor:

    if use_temperature and T != 1.0:
        student_out = student_out / T
        teacher_out = teacher_out.detach() / T

    soft_loss = F.mse_loss(student_out, teacher_out.detach())
    hard_loss = F.mse_loss(student_out * (T if use_temperature else 1.0), hard_target)

    return kd_ratio * soft_loss + (1 - kd_ratio) * hard_loss

KD_LOSSES = {"logits": kd_loss_regression}

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


@step(enable_cache=False)
def student_distiller(
    train: Dict[str, pd.DataFrame],
    val: Dict[str, pd.DataFrame],
    seq_len: int,
    horizon: int,
    selected_target_columns: List[str],
    teacher: nn.Module,
    teacher_hparams: Dict[str, Any],
    student_kind: StudentType,
    kd_kind: KDKind,
    alpha: float,
    T: float,
    use_temperature: bool,
    epochs: int,
    early_stop_epochs,
    batch,
    lr,
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
        target_cols=selected_target_columns,
    )
    val_loader, _ = make_loader(
        dfs=val,
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch,
        shuffle=False,
        target_cols=selected_target_columns,
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
        student.train()
        run_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            # forward teacher & student
            with torch.no_grad():
                t_out = teacher(X)                # (B, H, T)
            s_out = student(X)                    # (B, H, T)

            loss = kd_fn(student_out=s_out,
                         teacher_out=t_out,
                         hard_target=y,
                         alpha=alpha,
                         T=T,
                         use_temperature=use_temperature)

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
