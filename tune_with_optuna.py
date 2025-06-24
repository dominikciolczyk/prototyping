import optuna
from zenml.client import Client
from train_pipeline import training_pipeline

def objective(trial: optuna.Trial) -> float:
    threshold_strategy = trial.suggest_categorical("threshold_strategy", ["zscore", "iqr", "robust_zscore"])
    reduction_method = trial.suggest_categorical("reduction_method", ["interpolate_linear", "interpolate_polynomial"])
    scaler_method = trial.suggest_categorical("scaler_method", ["standard", "minmax"])
    basic = trial.suggest_categorical("basic", [True, False])
    cyclical = trial.suggest_categorical("cyclical", [True, False])
    is_weekend_mode = trial.suggest_categorical("is_weekend_mode", ["none", "numeric", "categorical", "both"])

    # DPSO-GA params
    swarm_size = trial.suggest_int("swarm_size", 10, 100)
    max_iters = trial.suggest_int("max_iters", 20, 200)
    inertia = trial.suggest_float("inertia", 0.1, 1.0)
    cognitive = trial.suggest_float("cognitive", 0.1, 2.0)
    social = trial.suggest_float("social", 0.1, 2.0)

    if is_weekend_mode == "categorical" and not basic:
        raise optuna.TrialPruned()

    # 2. Run a full ZenML pipeline with these hyperparameters
    training_pipeline(
        n_estimators=n_estimators,
        max_depth=max_depth
    )()

    # 3. Retrieve the last run’s accuracy
    client = Client()
    run = client.get_pipeline("training_pipeline").last_run
    accuracy = run.steps["train_step"].outputs["accuracy"][0]
    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best hyperparameters:", study.best_trial.params)
