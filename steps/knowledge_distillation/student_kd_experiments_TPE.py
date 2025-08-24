import json
import optuna
import pandas as pd
from torch import nn
from zenml import step
from zenml.logger import get_logger
from .student_distiller import student_distiller, StudentType
from steps.training import model_evaluator
from typing import Dict, Any, List

logger = get_logger(__name__)


@step(enable_cache=False)
def student_kd_optuna_experiments(
    train: Dict[str, pd.DataFrame],
    val: Dict[str, pd.DataFrame],
    test: Dict[str, pd.DataFrame],
    seq_len: int,
    horizon: int,
    selected_target_columns: List[str],
    teacher: nn.Module,
    teacher_hparams: Dict[str, Any],
    student_kind: StudentType,
    epochs: int,
    early_stop_epochs: int,
    scalers: Dict[str, Dict[str, Any]],
    eval_alpha: float,
    eval_beta: float,
    n_trials: int = 30,
) -> pd.DataFrame:
    """Run Optuna TPE search for KD + CNN + optimizer params."""

    results = []

    def objective(trial: optuna.Trial) -> float:
        kd_kind = trial.suggest_categorical("kd_kind", ["AsymmetricSmoothL1", "AsymmetricL1"])

        distill_alpha = trial.suggest_float("distill_alpha", 0.1, 0.9, step=0.2)
        cnn_kernels_options = {
            # --- pojedyncze ---
            "c16_k3": ([16], [3]),
            "c16_k5": ([16], [5]),
            "c16_k7": ([16], [7]),
            "c16_k9": ([16], [9]),
            "c16_k11": ([16], [11]),
            "c16_k13": ([16], [13]),

            "c32_k3": ([32], [3]),
            "c32_k5": ([32], [5]),
            "c32_k7": ([32], [7]),
            "c32_k9": ([32], [9]),
            "c32_k11": ([32], [11]),
            "c32_k13": ([32], [13]),

            "c64_k3": ([64], [3]),
            "c64_k5": ([64], [5]),
            "c64_k7": ([64], [7]),
            "c64_k9": ([64], [9]),
            "c64_k11": ([64], [11]),
            "c64_k13": ([64], [13]),

            "c128_k3": ([128], [3]),
            "c128_k5": ([128], [5]),
            "c128_k7": ([128], [7]),
            "c128_k9": ([128], [9]),
            "c128_k11": ([128], [11]),
            "c128_k13": ([128], [13]),

            "c256_k3": ([256], [3]),
            "c256_k5": ([256], [5]),
            "c256_k7": ([256], [7]),
            "c256_k9": ([256], [9]),
            "c256_k11": ([256], [11]),
            "c256_k13": ([256], [13]),

            "16_32_k3_5": ([16, 32], [3, 5]),
            "16_32_k5_7": ([16, 32], [5, 7]),
            "16_32_k7_9": ([16, 32], [7, 9]),
            "16_32_k9_11": ([16, 32], [9, 11]),
            "16_32_k11_13": ([16, 32], [11, 13]),

            "32_64_k3_5": ([32, 64], [3, 5]),
            "32_64_k5_7": ([32, 64], [5, 7]),
            "32_64_k7_9": ([32, 64], [7, 9]),
            "32_64_k9_11": ([32, 64], [9, 11]),
            "32_64_k11_13": ([32, 64], [11, 13]),

            "64_128_k3_5": ([64, 128], [3, 5]),
            "64_128_k5_7": ([64, 128], [5, 7]),
            "64_128_k7_9": ([64, 128], [7, 9]),
            "64_128_k9_11": ([64, 128], [9, 11]),
            "64_128_k11_13": ([64, 128], [11, 13]),

            "128_256_k3_5": ([128, 256], [3, 5]),
            "128_256_k5_7": ([128, 256], [5, 7]),
            "128_256_k7_9": ([128, 256], [7, 9]),
            "128_256_k9_11": ([128, 256], [9, 11]),
            "128_256_k11_13": ([128, 256], [11, 13]),

        }

        choice_key = trial.suggest_categorical("cnn_kernels", list(cnn_kernels_options.keys()))
        cnn_channels, kernels = cnn_kernels_options[choice_key]

        #lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
        lr = 1e-3
        batch = 64
        #batch = trial.suggest_categorical("batch", [32, 64, 128])
        lstm_hidden = trial.suggest_categorical("lstm_hidden", [32, 64, 128])
        #dropout = trial.suggest_float("dropout", 0.1, 0.3, step=0.1)
        dropout = 0.1

        alpha = trial.suggest_categorical("alpha", [5, 10]) if kd_kind in (
        "AsymmetricL1", "AsymmetricSmoothL1") else None
        beta = trial.suggest_categorical("beta", [1, 3]) if kd_kind == "AsymmetricSmoothL1" else None

        run_name = (
            f"{kd_kind}_distill={distill_alpha}_cnn={cnn_channels}"
            f"_kern={kernels}_lr={lr:.4f}_batch={batch}_lstm={lstm_hidden}_drop={dropout}"
            + (f"_alpha={alpha}" if alpha is not None else "")
            + (f"_beta={beta}" if beta is not None else "")
        )
        logger.info(f"🔎 Running: {run_name}")

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
            hyper_params={"batch": batch},
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
            "metrics": result["metrics"],
        }

        results.append(row)
        with open("student_kd_optuna_results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

        return score

    # TPE-based search
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)

    df = pd.DataFrame(results)
    best = study.best_trial.params
    logger.info(f"🏆 Best config: {best}")
    return df