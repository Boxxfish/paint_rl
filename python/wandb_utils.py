"""
Utilities for working with Weights and Biases.
"""

from typing import Dict
import wandb

def start_run(project: str, config: Dict[str, float]):
    """
    Begins logging a new run.
    """
    run = wandb.init(project=project, config=config)
    if run is None:
        raise RuntimeError("wandb.init did not return valid run object")
    return run


def log(run, values: dict[str, list[float | None]], step_count: int):
    """
    Logs a batch of data.
    Each element represents the metric for a single step.
    """
    for i in range(step_count):
        log_dict = {}
        for key in values:
            step_value = values[key][i]
            if step_value is not None:
                log_dict[key] = step_value
        run.log(log_dict)


def finish_run(run):
    """
    Finishes logging the current run.
    """
    run.finish()
