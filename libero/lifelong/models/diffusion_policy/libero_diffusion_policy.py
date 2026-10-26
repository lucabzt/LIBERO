import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.diffusion_policy.diffusion_unet_timm_policy import (
    DiffusionUnetTimmPolicy,
)
from libero.lifelong.models.diffusion_policy.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from libero.lifelong.models.diffusion_policy.timm_obs_encoder import TimmObsEncoder

TASK_EMB_KEY = "task_emb"


class DiffusionPolicy(BasePolicy):
    """
    LIBERO wrapper around the Memory Anchors DiffusionUnetTimmPolicy (self.dp).

    Training batches come from robomimic's SequenceDataset with
    frame_stack = n_obs_steps and seq_len = horizon, so obs hold
    n_obs_steps - 1 + horizon frames (t - n_obs_steps + 1 ... t + horizon - 1)
    and actions hold horizon steps (t ... t + horizon - 1). The policy is
    conditioned on the first n_obs_steps frames and predicts all actions.

    The language embedding is passed to the obs encoder as an extra low-dim
    obs key, so it is concatenated with the visual and proprio features.
    """

    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        assert cfg.data.frame_stack == policy_cfg.n_obs_steps, (
            "data.frame_stack must equal policy.n_obs_steps"
        )
        assert cfg.data.seq_len >= policy_cfg.horizon, (
            "data.seq_len must be at least policy.horizon"
        )
        self.n_obs_steps = policy_cfg.n_obs_steps
        self.horizon = policy_cfg.horizon
        self.n_action_steps = policy_cfg.n_action_steps
        self.rgb_keys = list(cfg.data.obs.modality.rgb)
        self.low_dim_keys = list(cfg.data.obs.modality.low_dim)

        obs_shape_meta = {}
        for key in self.rgb_keys:
            obs_shape_meta[key] = {
                "shape": list(shape_meta["all_shapes"][key]),
                "type": "rgb",
                "horizon": self.n_obs_steps,
            }
        for key in self.low_dim_keys:
            obs_shape_meta[key] = {
                "shape": list(shape_meta["all_shapes"][key]),
                "type": "low_dim",
                "horizon": self.n_obs_steps,
            }
        obs_shape_meta[TASK_EMB_KEY] = {
            "shape": [policy_cfg.language_encoder.network_kwargs.input_size],
            "type": "low_dim",
            "horizon": 1,
        }
        dp_shape_meta = {
            "obs": obs_shape_meta,
            "action": {
                "shape": [shape_meta["ac_dim"]],
                "horizon": self.horizon,
                "components": policy_cfg.action_components,
            },
        }

        noise_scheduler = DDIMScheduler(**policy_cfg.noise_scheduler)
        obs_encoder = TimmObsEncoder(shape_meta=dp_shape_meta, **policy_cfg.obs_encoder)
        self.dp = DiffusionUnetTimmPolicy(
            shape_meta=dp_shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            num_inference_steps=policy_cfg.num_inference_steps,
            **policy_cfg.unet,
        )
        self.dp.set_normalizer(self._build_normalizer(cfg))

        self.obs_history = []
        self.action_queue = []

    def _build_normalizer(self, cfg):
        """
        Images: [0, 1] -> [-1, 1]. Language embedding: identity.
        Actions and low-dim obs: min/max limits -> [-1, 1], from the stats main.py
        computes over the training data (cfg.normalizer_stats).
        """
        normalizer = LinearNormalizer()
        for key in self.rgb_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_manual(
                scale=torch.tensor([2.0]),
                offset=torch.tensor([-1.0]),
                input_stats_dict={
                    "min": torch.tensor([0.0]),
                    "max": torch.tensor([1.0]),
                    "mean": torch.tensor([0.5]),
                    "std": torch.tensor([0.5]),
                },
            )
        normalizer[TASK_EMB_KEY] = SingleFieldLinearNormalizer.create_identity()
        for key in self.low_dim_keys + ["action"]:
            stats = cfg.normalizer_stats[key]
            # fitting 'limits' on just the min and max rows gives the same scale/offset
            # as fitting on the full data (mean/std stats are unused)
            limits = torch.tensor([stats["min"], stats["max"]], dtype=torch.float32)
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(limits, mode="limits")
        return normalizer

    def _dp_obs(self, obs, task_emb):
        dp_obs = {key: obs[key] for key in self.rgb_keys + self.low_dim_keys}
        dp_obs[TASK_EMB_KEY] = task_emb.unsqueeze(1)
        return dp_obs

    def _dp_batch(self, data):
        obs = {
            key: data["obs"][key][:, : self.n_obs_steps]
            for key in self.rgb_keys + self.low_dim_keys
        }
        return {
            "obs": self._dp_obs(obs, data["task_emb"]),
            "action": data["actions"][:, : self.horizon],
        }

    def forward(self, data):
        # used by compute_flops; the diffusion training loss is the forward pass
        return self.compute_loss(data)

    def compute_loss(self, data, reduction="mean"):
        return self.dp.compute_loss(self._dp_batch(data), reduce_loss=(reduction == "mean"))

    @torch.no_grad()
    def get_action(self, data):
        """
        Called once per env step with obs of shape (B, ...). Keeps the last
        n_obs_steps obs (the first one repeated at episode start, like robomimic's
        frame stack padding), replans every n_action_steps steps.
        """
        self.eval()
        obs = {key: data["obs"][key] for key in self.rgb_keys + self.low_dim_keys}
        if len(self.obs_history) == 0:
            self.obs_history = [obs] * self.n_obs_steps
        else:
            self.obs_history = self.obs_history[1:] + [obs]

        if len(self.action_queue) == 0:
            stacked = {
                key: torch.stack([o[key] for o in self.obs_history], dim=1)
                for key in obs
            }
            actions = self.dp.predict_action(self._dp_obs(stacked, data["task_emb"]))["action"]
            self.action_queue = list(actions[:, : self.n_action_steps].unbind(dim=1))
        return self.action_queue.pop(0).detach().cpu().numpy()

    def reset(self):
        self.obs_history = []
        self.action_queue = []
