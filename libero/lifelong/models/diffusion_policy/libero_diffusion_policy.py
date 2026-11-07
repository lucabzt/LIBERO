import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.modules.language_modules import *
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
    obs key, so it is concatenated with the visual and proprio features. With
    policy.project_language it is first projected by policy.language_encoder
    (MLPEncoder, as in LIBERO's BC policies), trained with the policy.
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

        # default False: checkpoints from before this option have no projection
        self.language_encoder = None
        task_emb_size = policy_cfg.language_encoder.network_kwargs.input_size
        if policy_cfg.get("project_language", False):
            self.language_encoder = eval(policy_cfg.language_encoder.network)(
                **policy_cfg.language_encoder.network_kwargs
            )
            task_emb_size = policy_cfg.language_encoder.network_kwargs.output_size

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
            "shape": [task_emb_size],
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
        if self.language_encoder is not None:
            task_emb = self.language_encoder({"task_emb": task_emb})
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

    def freeze_obs_encoder(self):
        """
        Freeze the obs encoder (ResNets of both cameras and the spatial embedding; low-dim obs and
        the raw task embedding pass through it without parameters), e.g. after multitask
        pretraining: only the U-Net (and the language projection, if any) is trained afterwards.
        The encoder keeps its train-mode random crop on the current-task data.
        """
        for param in self.dp.obs_encoder.parameters():
            param.requires_grad = False

    def obs_encoder_frozen(self):
        return not any(p.requires_grad for p in self.dp.obs_encoder.parameters())

    @torch.no_grad()
    def encode_latents(self, data):
        """
        Latent replay entries of a training batch: (global condition (B, D) as the U-Net sees it,
        i.e. obs-encoder features of the normalized obs incl. the task embedding, and the
        unnormalized action chunk (B, horizon, action_dim)). Call in eval mode (center crop).
        """
        batch = self._dp_batch(data)
        nobs = self.dp.normalizer.normalize(batch["obs"])
        return self.dp.obs_encoder(nobs), batch["action"]

    def latent_dims(self):
        """{obs key: feature dims in the global condition}, in the order of the condition."""
        enc = self.dp.obs_encoder
        keys = enc.rgb_keys + enc.low_dim_keys
        meta = enc.shape_meta["obs"]
        example = {
            k: torch.zeros((1, meta[k]["horizon"]) + tuple(meta[k]["shape"]), device=enc.device)
            for k in keys
        }
        with torch.no_grad():
            return {k: int(enc.feature_by_key(example, k).shape[1]) for k in keys}

    def compute_loss_with_latent_replay(self, data, replay_cond, replay_actions):
        """
        Diffusion loss on a current-task batch (raw obs through the obs encoder) concatenated with
        a latent replay batch (stored global conditions and unnormalized action chunks).
        """
        batch = self._dp_batch(data)
        nobs = self.dp.normalizer.normalize(batch["obs"])
        global_cond = torch.cat([self.dp.obs_encoder(nobs), replay_cond], dim=0)
        actions = torch.cat([batch["action"], replay_actions], dim=0)
        nactions = self.dp.normalizer["action"].normalize(actions)
        return self.dp.compute_loss_from_global_cond(global_cond, nactions)

    @torch.no_grad()
    def compute_obs_representations(self, data):
        """
        Obs-encoder features of a training batch, {obs key: (B, D_key)}: the per-key parts
        of the global condition (rgb, low-dim and task embedding), used by AnchorER.
        """
        return self.dp.compute_obs_representations(self._dp_batch(data), return_as_dict=True)

    @torch.no_grad()
    def compute_action_disagreement(
        self, data, noise_level, ood_metric=None, num_inference_steps=None, action_scale=None
    ):
        """
        Noise-denoise action disagreement of a training batch, (B,): the action chunk is
        noised to step noise_level and denoised with the policy, then compared to the label
        (DiffusionUnetTimmPolicy.compute_ood_score, Memory Anchors Alg. 1 step 2).
        """
        return self.dp.compute_ood_score(
            self._dp_batch(data),
            noise_level=noise_level,
            ood_metric=ood_metric,
            num_inference_steps=num_inference_steps,
            action_scale=action_scale,
        )

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
