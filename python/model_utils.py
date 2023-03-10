"""
Utilities for working with PyTorch models within a Rust project.
"""
import torch
import importlib

def export_model(file_name: str, model_name: str, params: object, input_shapes: list[list[int]], cuda: bool = False):
    """
    Exports a model from a file, using `input_shape` as input.
    """
    device = torch.device("cuda" if cuda else "cpu", 0)
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
    net.to(device)
    fake_input = [torch.rand(input_shape, device=device) for input_shape in input_shapes]

    # Test model runs with fake input
    net.eval()
    net(*fake_input)
    net.train()
    net(*fake_input)

    # Trace model
    traced = torch.jit.script(net, fake_input)
    traced.save(f"temp/{model_name}.ptc")


def get_img_size(old_img_size: int, conv: torch.nn.Conv2d) -> int:
    """
    Returns the size of the image after the convolution is run on it.
    """
    return (old_img_size + 2 * int(conv.padding[0]) - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1

def init_orthogonal(module: torch.nn.Module):
    """
    Initializes the weights of the model so they're orthogonal.
    """
    for param in module.parameters():
        if len(param.shape) > 1:
            torch.nn.init.orthogonal_(param)