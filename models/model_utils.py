"""
Utilities for working with PyTorch models within a Rust project.
"""
import torch
import importlib

def export_model(file_name: str, model_name: str, params: object, input_shape: list[int]):
    """
    Exports a model from a file, using `input_shape` as tracing input.
    """
    module = importlib.import_module("models." + file_name)
    model = getattr(module, model_name)
    if isinstance(params, dict):
        net = model(**params)
    else:
        d = {}
        for key in dir(params):
            if "__" not in key:
                d[key] = getattr(params, key)
        net = model(**d)
    fake_input = torch.zeros(input_shape)
    traced = torch.jit.script(net, fake_input)
    traced.save(f"temp/{model_name}.ptc")