import inspect

import torch

# weights_only only exists since PyTorch 1.13; since 2.6 it defaults to True, which cannot
# load LIBERO's init states and checkpoints. Pass it only where torch.load supports it.
_WEIGHTS_ONLY_KWARGS = (
    {"weights_only": False}
    if "weights_only" in inspect.signature(torch.load).parameters
    else {}
)


def torch_load(*args, **kwargs):
    return torch.load(*args, **_WEIGHTS_ONLY_KWARGS, **kwargs)
