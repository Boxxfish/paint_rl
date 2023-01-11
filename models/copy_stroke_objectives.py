"""
Objective functions for the copy stroke environment.
"""

def greatest_eval_reward(params: dict[str, float]) -> float:
    return -params["eval_reward"]