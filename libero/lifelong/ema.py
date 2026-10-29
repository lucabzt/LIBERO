import copy

import torch


class EMA:
    """
    Exponential moving average of a model's weights, as EMAModel of the diffusion_policy /
    UMI training workspace: after every optimizer step
        averaged = decay * averaged + (1 - decay) * weights,  decay = 1 - (1 + step)^-power,
    clipped to max_value. Parameters without gradients (e.g. the normalizer) and buffers are
    copied instead of averaged.
    """

    def __init__(self, model, power=0.75, max_value=0.9999):
        self.model = model
        self.power = power
        self.max_value = max_value
        self.optimization_step = 0
        self.averaged = copy.deepcopy(model.state_dict())
        self.trainable = {n for n, p in model.named_parameters() if p.requires_grad}

    def get_decay(self):
        # as EMAModel.get_decay with update_after_step=0, inv_gamma=1, min_value=0
        step = self.optimization_step - 1
        if step <= 0:
            return 0.0
        return min(1 - (1 + step) ** -self.power, self.max_value)

    @torch.no_grad()
    def step(self):
        self.optimization_step += 1
        decay = self.get_decay()
        for name, value in self.model.state_dict().items():
            if name in self.trainable:
                self.averaged[name].mul_(decay).add_(value, alpha=1 - decay)
            else:
                self.averaged[name].copy_(value)

    def copy_to_model(self):
        self.model.load_state_dict(self.averaged)
