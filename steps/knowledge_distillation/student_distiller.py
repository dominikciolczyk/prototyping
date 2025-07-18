from typing import Dict, Any, List, Literal, Tuple, Callable
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
from losses.qos import AsymmetricSmoothL1, AsymmetricL1
from zenml.logger import get_logger

logger = get_logger(__name__)

StudentType   = Literal["cnn_lstm", "lstm"]
KDKind = Literal["mse", "AsymmetricSmoothL1", "AsymmetricL1"]

def make_distill_loss(
    base_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    distill_alpha: float
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Creates a KD loss: alpha * loss(student, teacher) + (1 - alpha) * loss(student, ground_truth)
    """
    def fn(student_out: torch.Tensor,
           teacher_out: torch.Tensor,
           ground_truth: torch.Tensor) -> torch.Tensor:
        loss_soft = base_loss_fn(student_out, teacher_out.detach())
        loss_hard = base_loss_fn(student_out, ground_truth)
        return distill_alpha * loss_soft + (1 - distill_alpha) * loss_hard
    return fn

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
    hard_loss = F.mse_loss(student_out, hard_target)

    return kd_ratio * soft_loss + (1 - kd_ratio) * hard_loss

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


@step(enable_cache=True)
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
    kd_params: Dict[str, float],
    epochs: int,
    early_stop_epochs,
    batch,
    lr,
) -> nn.Module:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device).eval()                         # freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False

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
    logger.info(f"Student_distiller | X {sample_X.shape}, y {sample_y.shape}")

    # ── 2. Build student ───────────────────────────────────────────────────
    student = build_student(student_kind,
                            n_features, n_targets, horizon,
                            teacher_hparams).to(device)
    optimizer = Adam(student.parameters(), lr=lr)

    if kd_kind == "mse":
        kd_fn = make_distill_loss(F.mse_loss, distill_alpha=kd_params["distill_alpha"])
        logger.info(f"Using MSE distillation with alpha={kd_params['distill_alpha']}")
    elif kd_kind == "AsymmetricSmoothL1":
        base_loss = AsymmetricSmoothL1(alpha=kd_params["alpha"], beta=kd_params["beta"])
        kd_fn = make_distill_loss(base_loss, distill_alpha=kd_params["distill_alpha"])
        logger.info(f"Using AsymmetricSmoothL1 distillation with "
                    f"distill_alpha={kd_params['distill_alpha']}, "
                    f"alpha={kd_params['alpha']}, beta={kd_params['beta']}")
    elif kd_kind == "AsymmetricL1":
        base_loss = AsymmetricL1(alpha=kd_params["alpha"])
        kd_fn = make_distill_loss(base_loss, distill_alpha=kd_params["distill_alpha"])
        logger.info(f"Using AsymmetricL1 distillation with "
                    f"distill_alpha={kd_params['distill_alpha']}, "
                    f"alpha={kd_params['alpha']}")
    else:
        raise ValueError(kd_kind)

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

            loss = kd_fn(s_out, t_out, y)

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

        # early-stopping logic
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(student.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_epochs:
                logger.info("Early-stopping triggered.")
                break

        logger.info(f"[{epoch:02d}] distill "
                    f"train={train_loss:.4f}  val={val_loss:.4f}  "
                    f"(patience {patience}/{early_stop_epochs})")

    # ── 4. Restore best weights ───────────────────────────────────────────
    if best_state:
        student.load_state_dict(best_state)
    else:
        logger.warning("No improvement – final weights kept.")

    logger.info(f"✅ Student ({student_kind}, {kd_kind}) best val {best_val:.4f}")
    return student
