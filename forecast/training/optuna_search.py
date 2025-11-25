import optuna
from typing import Callable, Dict


def run_optuna_search(
    objective_func: Callable[[optuna.Trial], float],
    n_trials: int = 50,
    study_name: str = "hyperparam_search",
    direction: str = "minimize",
    storage: str = None,
):
    """
    objective_func(trial) must return validation loss.
    """
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=True if storage else False,
    )
    study.optimize(objective_func, n_trials=n_trials)
    return study
