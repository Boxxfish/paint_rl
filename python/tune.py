"""
Optuna integration for experiments.
Optuna uses the arguments sent to an experiment to tune hyperparameters,
and uses the values returned to compute a loss to optimize.
"""

import optuna
import importlib
from collections.abc import Callable
from argparse import ArgumentParser
import subprocess
import re

def run_experiment(name: str, params: dict[str, float | int]) -> dict[str, float]:
    """
    Runs an experiment, passing the given params to the program.
    Returns a dict of final metrics.
    """
    cmd = ["cargo", "run", "--bin", name, "--release", "--"]
    for param_name in params:
        cmd.append("--" + param_name)
        cmd.append(str(params[param_name]))
    output = subprocess.run(cmd, capture_output=True)
    r = re.compile("[A-z]\\S+: -?\\d*\\.?\\d*")
    metrics = r.findall(output.stdout.decode("UTF8"))
    final_metrics = {}
    for metric in metrics:
        metric_name = metric.split(": ")[0]
        metric_val = float(metric.split(": ")[1])
        final_metrics[metric_name] = metric_val
    return final_metrics

def create_obj_wrapper(experiment_name: str, params: dict[str, float | int], obj_func: Callable[[dict[str, float]], float]) -> Callable[[optuna.Trial], float]:
    """
    Returns an objective function that launches an experiment with tuned hyperparameters,
    using the given base objective function.
    """
    def new_obj_func(trial: optuna.Trial) -> float:
        # Create dict of hyperparameters
        param_dict = {}
        for param_key in params:
            # Generate tuning ranges
            if param_key[-5:] == "__min":
                base_param = param_key[:-5]
                param_min = params[base_param + "__min"]
                param_max = params[base_param + "__max"]
                param_dict[base_param] = trial.suggest_float(base_param, param_min, param_max)
            # Any parameters without "__" should be passed in like normal
            if "__" not in param_key:
                param_dict[param_key] = params[param_key]
        metrics = run_experiment(experiment_name, param_dict)
        return obj_func(metrics)
    return new_obj_func
    

def run_study(name: str, objective_func: str, params: dict[str, float], trials: int):
    """
    Runs a study.
    Objective function should be passed in as "file_name.func_name".
    At the end, the best hyper parameters are printed to the console.
    
    If the params "param__min" and "param__max" are given, the tuner will use that as
    the range.
    """
    study = optuna.create_study()
    module = objective_func.split(".")[0]
    func_name = objective_func.split(".")[1]
    objective = getattr(importlib.import_module(module), func_name)
    obj_wrapper = create_obj_wrapper(name, params, objective)
    study.optimize(obj_wrapper, n_trials=trials)
    print(study.best_params)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--trials", required=True)
    args, params = parser.parse_known_args()
    params.pop(0)
    params_dict = {}
    for i in range(len(params) // 2):
        param_name = params[i * 2][2:]
        param_val_str = params[i * 2 + 1]
        param_val = float(param_val_str) if "." in param_val_str else int(param_val_str)
        params_dict[param_name] = param_val
    run_study(args.name, args.objective, params_dict, int(args.trials)) # type: ignore
