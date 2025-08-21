import json
import pandas as pd
from torch import nn
from zenml import step
import itertools
from .student_distiller import student_distiller, StudentType
from steps.training import model_evaluator
from typing import Dict, Any, List
from zenml.logger import get_logger

logger = get_logger(__name__)

@step(enable_cache=False)
def student_kd_experiments(
    train: Dict[str, pd.DataFrame],
    val: Dict[str, pd.DataFrame],
    test: Dict[str, pd.DataFrame],
    seq_len: int,
    horizon: int,
    selected_target_columns: List[str],
    teacher: nn.Module,
    student_kind: StudentType,
    epochs: int,
    early_stop_epochs: int,
    alpha_range: List[int],
    beta_range: List[int],
    distill_alpha_range: List[float],
    cnn_channels_grid: List[List[int]],
    kernels_grid: List[List[int]],
    lr_grid: List[float],
    batch_grid: List[int],
    lstm_hidden_grid: List[int],
    dropout_grid: List[float],
    scalers: Dict[str, Dict[str, Any]],
    eval_alpha: float,
    eval_beta: float,
) -> pd.DataFrame:
    """Run grid search for KD + CNN + optimizer params. Returns a DataFrame of all runs."""

    results = []

    logger.info(itertools.product(
                distill_alpha_range,
                cnn_channels_grid,
                kernels_grid,
                lr_grid,
                batch_grid,
                lstm_hidden_grid,
                dropout_grid,
        ))

    #for kd_kind in ["AsymmetricSmoothL1", "AsymmetricL1"]:
    for kd_kind in ["AsymmetricSmoothL1"]:
        for distill_alpha, cnn_channels, kernels, lr, batch, lstm_hidden, dropout in itertools.product(
                distill_alpha_range,
                cnn_channels_grid,
                kernels_grid,
                lr_grid,
                batch_grid,
                lstm_hidden_grid,
                dropout_grid,
        ):
            for alpha in (alpha_range if kd_kind in ("AsymmetricL1", "AsymmetricSmoothL1") else [None]):
                for beta in (beta_range if kd_kind == "AsymmetricSmoothL1" else [None]):

                    cnn_str = "x".join(map(str, cnn_channels))
                    kern_str = "x".join(map(str, kernels))

                    run_name = (
                            f"{kd_kind}"
                            f"_distill={distill_alpha}"
                            f"_cnn={cnn_str}"
                            f"_kern={kern_str}"
                            f"_lr={lr}"
                            f"_batch={batch}"
                            f"_lstm={lstm_hidden}"
                            f"_drop={dropout}"
                            + (f"_alpha={alpha}" if alpha is not None else "")
                            + (f"_beta={beta}" if beta is not None else "")
                    )
                    logger.info(f"🔎 Running: {run_name}")


                    # train student
                    student = student_distiller(
                        train=train,
                        val=val,
                        seq_len=seq_len,
                        horizon=horizon,
                        selected_target_columns=selected_target_columns,
                        teacher=teacher,
                        student_kind=student_kind,
                        kd_kind=kd_kind,
                        kd_params={
                            "distill_alpha": distill_alpha,
                            **({"alpha": alpha} if alpha is not None else {}),
                            **({"beta": beta} if beta is not None else {}),
                        },
                        epochs=epochs,
                        early_stop_epochs=early_stop_epochs,
                        batch=batch,
                        lr=lr,
                        cnn_channels=cnn_channels,
                        kernels=kernels,
                        lstm_hidden=lstm_hidden,
                        dropout=dropout,
                    )

                    result = model_evaluator.entrypoint(
                        model=student,
                        test=test,
                        seq_len=seq_len,
                        horizon=horizon,
                        alpha=eval_alpha,
                        beta=eval_beta,
                        hyper_params={
                            "batch": batch,
                        },
                        selected_target_columns=selected_target_columns,
                        scalers=scalers,
                    )

                    score = result["metrics"]["AsymmetricSmoothL1_model"]

                    row = {
                        "run": run_name,
                        "kd_kind": kd_kind,
                        "distill_alpha": distill_alpha,
                        "alpha": alpha,
                        "beta": beta,
                        "cnn_channels": cnn_channels,
                        "kernels": kernels,
                        "lr": lr,
                        "batch": batch,
                        "dropout": dropout,
                        "lstm_hidden": lstm_hidden,
                        "score": score,
                        "metrics": result["metrics"],  # optional: keep full metrics
                    }

                    results.append(row)

                    # write only this row to file
                    with open("student_kd_results.jsonl", "a") as f:
                        f.write(json.dumps(row) + "\n")

    df = pd.DataFrame(results)
    best = df.loc[df["score"].idxmin()]
    logger.info(f"🏆 Best config: {best.to_dict()}")
    return df
