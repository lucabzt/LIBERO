import torch
import torch.nn as nn


class ParameterDict(nn.Module):
    """
    nn.ParameterDict as of torch >= 1.12, which the normalizer relies on: values may be
    tensors (stored as frozen Parameters) or nested ParameterDicts. torch 1.11's version
    only accepts Parameters. State dict keys are the same as with torch >= 1.12.
    """

    def __init__(self, values=None):
        super().__init__()
        for key, value in (values or {}).items():
            self[key] = value

    def __setitem__(self, key, value):
        self._parameters.pop(key, None)
        self._modules.pop(key, None)
        if isinstance(value, nn.Module):
            self.add_module(key, value)
        else:
            if not isinstance(value, nn.Parameter):
                value = nn.Parameter(value, requires_grad=False)
            self.register_parameter(key, value)

    def __getitem__(self, key):
        if key in self._parameters:
            return self._parameters[key]
        return self._modules[key]

    def __contains__(self, key):
        return key in self._parameters or key in self._modules

    def __len__(self):
        return len(self._parameters) + len(self._modules)

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        return list(self._parameters.keys()) + list(self._modules.keys())

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def values(self):
        return [self[key] for key in self.keys()]


class DictOfTensorMixin(nn.Module):
    def __init__(self, params_dict=None):
        super().__init__()
        if params_dict is None:
            params_dict = ParameterDict()
        self.params_dict = params_dict

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        def dfs_add(dest, keys, value: torch.Tensor):
            if len(keys) == 1:
                dest[keys[0]] = value
                return

            if keys[0] not in dest:
                dest[keys[0]] = ParameterDict()
            dfs_add(dest[keys[0]], keys[1:], value)

        def load_dict(state_dict, prefix):
            out_dict = ParameterDict()
            for key, value in state_dict.items():
                value: torch.Tensor
                if key.startswith(prefix):
                    param_keys = key[len(prefix):].split('.')[1:]
                    # if len(param_keys) == 0:
                    #     import pdb; pdb.set_trace()
                    dfs_add(out_dict, param_keys, value.clone())
            return out_dict

        self.params_dict = load_dict(state_dict, prefix + 'params_dict')
        self.params_dict.requires_grad_(False)
        return 
