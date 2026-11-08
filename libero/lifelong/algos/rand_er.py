import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, RandomSampler, Sampler, Subset

from libero.lifelong.algos.base import Sequential
from libero.lifelong.algos.er import ER, cycle


class EndlessRandomSampler(Sampler):
    """Random permutations of range(n), one after another without end: the DataLoader never
    restarts (cycle() restarts it after every pass over the buffer, and with a small buffer and a
    large replay batch every restart stalls the training step), and every batch is full."""

    def __init__(self, n, seed):
        self.n = n
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        while True:
            yield from torch.randperm(self.n, generator=self.generator).tolist()


class FirstN(Dataset):
    """The first n samples of a task dataset (LIBERO ER's buffer: TruncatedSequenceDataset), keeping
    .sequence_dataset so that AnchorER can map indices to demos / timesteps."""

    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = min(n, len(dataset))
        self.sequence_dataset = dataset.sequence_dataset

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.dataset[idx]


class RandER(ER):
    """
    Experience replay with a buffer of a fixed fraction of all past task data, as
    RandER in the Memory Anchors paper (Sec. 3.1, App. B.3): the buffer holds
    buffer_fraction * (number of past transitions) transitions, sampled uniformly
    without replacement from all past tasks, and is resampled at every new task.
    lifelong.n_memories (if set) is a fixed buffer size instead.
    lifelong.buffer_indices_file (if set): a .npy of indices into the past data (all past tasks
    concatenated, as ConcatDataset) used as the buffer instead of select_buffer, e.g. for
    controlled single-task interventions with a fixed, precomputed buffer.

    Unlike ER (the first n_memories sequences of every past task), the buffer size
    grows with the amount of past data and is not split evenly across tasks.
    Every training batch is concatenated with a buffer batch of replay_ratio times its size
    (1.0: the same size, as ER).
    """

    def start_task(self, task):
        # skip ER.start_task, which builds the truncated per-task buffers
        Sequential.start_task(self, task)
        if self.current_task > 0:
            past_data = self.past_pool()
            n_memories = self.cfg.lifelong.get("n_memories")
            if n_memories is None:
                n_memories = max(1, round(self.cfg.lifelong.buffer_fraction * len(past_data)))
            assert n_memories <= len(past_data), "[error] buffer larger than the past data"
            rng = np.random.default_rng(self.cfg.seed + task)
            indices_file = self.cfg.lifelong.get("buffer_indices_file")
            if indices_file:
                indices = np.load(indices_file).astype(np.int64)
                assert len(np.unique(indices)) == len(indices) and indices.min() >= 0 and indices.max() < len(past_data), (
                    f"[error] invalid buffer indices in {indices_file}"
                )
                n_memories = len(indices)
                print(f"[info] buffer indices from {indices_file}")
            else:
                indices = self.select_buffer(past_data, n_memories, rng)
            buf = Subset(past_data, indices.tolist())
            self.replay_cost = {"n_buffer": int(n_memories), "n_past": int(len(past_data))}
            if self.cfg.lifelong.get("latent_replay", False):
                self._build_latent_buffer(buf, task)
                return
            self.replay_cost.update(self._buffer_bytes(raw=True))
            print(
                f"[info] {self.cfg.lifelong.algo} buffer: {n_memories} of {len(past_data)} "
                f"past transitions ({n_memories / len(past_data):.2%})"
            )
            replay_batch_size = max(
                1, round(self.cfg.lifelong.get("replay_ratio", 1.0) * self.cfg.train.batch_size)
            )
            if self.cfg.lifelong.get("continuous_replay_loader", False):
                self.buffer = iter(
                    DataLoader(
                        buf,
                        batch_size=replay_batch_size,
                        num_workers=self.cfg.train.num_workers,
                        sampler=EndlessRandomSampler(len(buf), self.cfg.seed + task),
                    )
                )
            else:
                self.buffer = cycle(
                    DataLoader(
                        buf,
                        batch_size=replay_batch_size,
                        num_workers=self.cfg.train.num_workers,
                        sampler=RandomSampler(buf),
                        persistent_workers=True,
                    )
                )

    def past_pool(self):
        """The past data the buffer is drawn from. lifelong.pool_n_per_task (if set): only the first
        pool_n_per_task samples of every past task, as LIBERO ER (n_memories=1000); with
        buffer_fraction=1.0 the buffer is then exactly ER's buffer."""
        n = self.cfg.lifelong.get("pool_n_per_task")
        if n is None:
            return ConcatDataset(self.datasets)
        return ConcatDataset([FirstN(ds, int(n)) for ds in self.datasets])

    def select_buffer(self, past_data, n_memories, rng):
        """Indices into past_data (ConcatDataset of the past tasks) of the replay buffer."""
        return rng.choice(len(past_data), size=n_memories, replace=False)

    # ------------------------------------------------------------------
    # Latent replay (MLR, Yu et al. 2026): with a frozen obs encoder the buffer holds the encoder
    # features (the U-Net's global condition) and action chunks instead of raw observations.
    # Since the encoder is frozen, encoding the selected samples at the start of the task gives
    # exactly the latents stored when the samples were first seen.
    # ------------------------------------------------------------------
    def _buffer_bytes(self, raw):
        """Storage per buffer sample and in total, raw (uint8 images of the n_obs_steps frames the
        policy reads, float32 low-dim obs and actions) and latent (float32 features without the
        task embedding, which is one vector per task, and float32 actions)."""
        policy = self.policy
        meta = self.cfg.shape_meta["all_shapes"]
        n_obs = self.cfg.policy.n_obs_steps
        action_bytes = 4 * self.cfg.policy.horizon * self.cfg.shape_meta["ac_dim"]
        raw_bytes = action_bytes + n_obs * (
            sum(int(np.prod(meta[k])) for k in self.cfg.data.obs.modality.rgb)
            + 4 * sum(int(np.prod(meta[k])) for k in self.cfg.data.obs.modality.low_dim)
        )
        dims = policy.latent_dims()
        latent_bytes = action_bytes + 4 * sum(d for k, d in dims.items() if k != "task_emb")
        n = self.replay_cost["n_buffer"]
        return {
            "replay": "raw" if raw else "latent",
            "raw_bytes_per_sample": int(raw_bytes),
            "latent_bytes_per_sample": int(latent_bytes),
            "buffer_bytes": int(n * (raw_bytes if raw else latent_bytes)),
            "buffer_bytes_if_raw": int(n * raw_bytes),
            "buffer_bytes_if_latent": int(n * latent_bytes),
        }

    def _build_latent_buffer(self, buf, task):
        assert self.policy.obs_encoder_frozen(), (
            "[error] lifelong.latent_replay needs a frozen obs encoder (policy.freeze_obs_encoder)"
        )
        t0 = time.time()
        loader = DataLoader(buf, batch_size=128, shuffle=False, num_workers=self.cfg.train.num_workers)
        conds, actions = [], []
        was_training = self.policy.training
        self.policy.eval()  # center crop, as at test time
        for data in loader:
            cond, act = self.policy.encode_latents(self.map_tensor_to_device(data))
            conds.append(cond)
            actions.append(act)
        self.policy.train(was_training)
        self.latent_cond = torch.cat(conds).contiguous()
        self.latent_actions = torch.cat(actions).contiguous()
        self.latent_generator = torch.Generator(device=self.latent_cond.device)
        self.latent_generator.manual_seed(self.cfg.seed + task)
        self.latent_perm, self.latent_pos = None, 0
        self.replay_batch_size = max(
            1, round(self.cfg.lifelong.get("replay_ratio", 1.0) * self.cfg.train.batch_size)
        )
        self.buffer = None  # observe takes the latent path
        self.replay_cost.update(self._buffer_bytes(raw=False))
        self.replay_cost["latent_encode_time_s"] = time.time() - t0
        self.replay_cost["latent_dim"] = int(self.latent_cond.shape[1])
        print(
            f"[info] {self.cfg.lifelong.algo} latent buffer: {len(self.latent_cond)} samples, "
            f"global cond dim {self.latent_cond.shape[1]}, "
            f"{self.replay_cost['buffer_bytes'] / 1e6:.1f} MB latent vs "
            f"{self.replay_cost['buffer_bytes_if_raw'] / 1e6:.1f} MB raw, "
            f"encoded in {self.replay_cost['latent_encode_time_s']:.0f} s"
        )

    def _next_latent_indices(self):
        """Endless random permutations of the latent buffer (as continuous_replay_loader)."""
        n, out, need = len(self.latent_cond), [], self.replay_batch_size
        while need > 0:
            if self.latent_perm is None or self.latent_pos >= n:
                self.latent_perm = torch.randperm(
                    n, generator=self.latent_generator, device=self.latent_cond.device
                )
                self.latent_pos = 0
            take = self.latent_perm[self.latent_pos : self.latent_pos + need]
            self.latent_pos += len(take)
            need -= len(take)
            out.append(take)
        return torch.cat(out)

    def observe(self, data):
        if getattr(self, "latent_cond", None) is None or self.current_task == 0:
            return super().observe(data)
        data = self.map_tensor_to_device(data)
        idx = self._next_latent_indices()
        self.optimizer.zero_grad()
        loss = self.policy.compute_loss_with_latent_replay(
            data, self.latent_cond[idx], self.latent_actions[idx]
        )
        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_norm_(
                [p for p in self.policy.parameters() if p.requires_grad], self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return loss.item()
