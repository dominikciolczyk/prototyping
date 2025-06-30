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
def student_kd_experiments(
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
    batch: int,
    lr: float,
    n_trials: int,
    alpha_range: List[int],
    beta_range: List[int],
    distill_alpha_range: List[float],
    scalers: Dict[str, Dict[str, Any]],
    eval_alpha: float,
    eval_beta: float,
) -> pd.DataFrame:
    """Run Optuna to find best KD hyperparameters, return a DataFrame of all trials."""
    # objective for Optuna
    def objective(trial: optuna.Trial) -> float:
        # sample the KD variant
        kd_kind = trial.suggest_categorical("kd_kind",
                                            ["mse", "AsymmetricL1", "AsymmetricSmoothL1"])
        # always sample distill_alpha
        distill_alpha = trial.suggest_categorical("distill_alpha", distill_alpha_range)

        # conditionally sample alpha and beta
        alpha = None
        beta  = None
        if kd_kind in ("AsymmetricL1", "AsymmetricSmoothL1"):
            alpha = trial.suggest_categorical("alpha", alpha_range)
        if kd_kind == "AsymmetricSmoothL1":
            beta = trial.suggest_categorical("beta", beta_range)

        run_name = f"{kd_kind}_dα={distill_alpha}" + (
            f"_α={alpha}" if alpha is not None else ""
        ) + (
            f"_β={beta}"  if beta  is not None else ""
        )
        logger.info(f"🔎 Trial {trial.number}: {run_name}")

        # call your existing distiller step
        student = student_distiller(
            train=train,
            val=val,
            seq_len=seq_len,
            horizon=horizon,
            selected_target_columns=selected_target_columns,
            teacher=teacher,
            teacher_hparams=teacher_hparams,
            student_kind=student_kind,
            kd_kind=kd_kind,
            kd_params={
                "distill_alpha": distill_alpha,
                **({"alpha": alpha} if alpha is not None else {}),
                **({"beta": beta}  if beta  is not None else {}),
            },
            epochs=epochs,
            early_stop_epochs=early_stop_epochs,
            batch=batch,
            lr=lr,
        )

        # evaluate on test set
        result = model_evaluator.entrypoint(
            model=student,
            test=test,
            seq_len=seq_len,
            horizon=horizon,
            alpha=eval_alpha,
            beta=eval_beta,
            hyper_params=teacher_hparams,
            selected_target_columns=selected_target_columns,
            scalers=scalers,
        )

        score = result["metrics"]["AsymmetricSmoothL1_model"]
        # report to Optuna (no intermediate pruning here, but you could)
        trial.report(score, step=0)
        return score

    # create and run the study
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)

    # build a DataFrame of all trials for your downstream report
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    logger.info(f"🏆 Best trial: #{study.best_trial.number} → "
                f"value={study.best_value:.4f}, params={study.best_trial.params}")

    return df