import os
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


def proportional_quotas(sizes, n):
    """n split over groups proportionally to sizes, rounded by largest remainder (sum exactly n)."""
    sizes = np.asarray(sizes, dtype=np.int64)
    exact = n * sizes / sizes.sum()
    quotas = np.floor(exact).astype(np.int64)
    rest = n - quotas.sum()
    quotas[np.argsort(-(exact - quotas), kind="stable")[:rest]] += 1
    assert quotas.sum() == n and (quotas <= sizes).all()
    return quotas


def equal_quotas(capacities, n, rng):
    """n split as equally as possible over groups with the given capacities: round robin in a
    random group order, groups that are full drop out (their spare share goes to the others)."""
    capacities = np.asarray(capacities, dtype=np.int64)
    assert capacities.sum() >= n
    alloc = np.zeros_like(capacities)
    order = rng.permutation(len(capacities))
    while alloc.sum() < n:
        for g in order:
            if alloc.sum() == n:
                break
            if alloc[g] < capacities[g]:
                alloc[g] += 1
    return alloc


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

    lifelong.buffer_selection: how the n_memories samples are chosen from the past data
      uniform:         uniformly from all past data (RandER; per-task counts vary randomly)
      task_stratified: exact per-task quotas proportional to the task sizes (largest remainder),
                       uniformly within each task (control: RandER without the count noise)
      rainbow:         the same quotas; within each task the samples are sorted by the current
                       policy's noise-denoise action disagreement (AnchorER's step-2 score) and
                       taken at evenly spaced ranks, so the buffer spans easy to hard samples
                       (Rainbow Memory, Bang et al. CVPR 2021, with the disagreement in place of
                       the perturbation uncertainty)
      kmeans:          the same quotas; within each task k-means (kmeans_n_clusters) on the current
                       policy's obs latent (AnchorER representation_keys, every dimension z-scored
                       with the past-data stats, as AnchorER step 1), the task's quota split equally
                       over its clusters (spare slots of small clusters go to the others), random
                       within a cluster: equal coverage of the task's state space

    lifelong.fixed_buffer: instead of reselecting the whole buffer at every task, every task's
    share is selected once (with buffer_selection, from that task's data only) at the start of the
    next task and never changed (see _fixed_buffer): no past data has to be stored beyond the buffer.

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
            elif self.cfg.lifelong.get("fixed_buffer", False):
                indices = self._fixed_buffer(past_data, rng)
                n_memories = len(indices)
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
        mode = self.cfg.lifelong.get("buffer_selection", "uniform")
        if mode == "uniform":
            return rng.choice(len(past_data), size=n_memories, replace=False)
        sizes = np.diff([0] + list(past_data.cumulative_sizes))
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        quotas = proportional_quotas(sizes, n_memories)
        if mode == "task_stratified":
            parts = [off + rng.choice(n, size=q, replace=False) for off, n, q in zip(offsets, sizes, quotas)]
            print(f"[info] task-stratified buffer, per task {quotas.tolist()}")
            return np.concatenate(parts)
        if mode == "rainbow":
            return self._rainbow_selection(past_data, sizes, offsets, quotas)
        if mode == "kmeans":
            return self._kmeans_selection(past_data, sizes, offsets, quotas, rng)
        raise ValueError(f"[error] unknown lifelong.buffer_selection {mode}")

    def _fixed_buffer(self, past_data, rng):
        """lifelong.fixed_buffer: the share of every past task is chosen once, at the start of the
        next task, from that task's data only (with the policy right after that task; for kmeans
        the latents are z-scored with that task's own stats) and kept unchanged afterwards, so the
        data of a task can be dropped once its share is stored. Share of task k:
        max(1, round(buffer_fraction * n_k)) (as the proportional quotas of the reselected buffer up
        to rounding). Every share is saved as local indices in <run>/fixed_buffer/task<k>.npy and
        loaded from there (resume, later tasks). Returns the indices into past_data."""
        cfg = self.cfg.lifelong
        assert cfg.get("n_memories") is None, "[error] lifelong.fixed_buffer uses buffer_fraction per task"
        assert not cfg.get("buffer_indices_file"), "[error] lifelong.fixed_buffer with buffer_indices_file"
        mode = cfg.get("buffer_selection", "uniform")
        sizes = np.diff([0] + list(past_data.cumulative_sizes))
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        out_dir = os.path.join(self.experiment_dir, "fixed_buffer")
        os.makedirs(out_dir, exist_ok=True)
        parts, quotas = [], []
        for k, (off, n) in enumerate(zip(offsets, sizes)):
            path = os.path.join(out_dir, f"task{k}.npy")
            if not os.path.exists(path):
                # only the task that just ended can be selected now (its data and the policy after it)
                assert k == self.current_task - 1, f"[error] fixed buffer share of task {k} missing: {path}"
                q = max(1, round(cfg.buffer_fraction * n))
                task_data = Subset(past_data, list(range(off, off + n)))
                if mode in ("uniform", "task_stratified"):
                    chosen = np.sort(rng.choice(n, size=q, replace=False))
                elif mode == "kmeans":
                    chosen = self._kmeans_selection(task_data, np.array([n]), np.array([0]), np.array([q]), rng, tag=f"fixed_task{k}")
                elif mode == "rainbow":
                    chosen = self._rainbow_selection(task_data, np.array([n]), np.array([0]), np.array([q]), tag=f"fixed_task{k}")
                else:
                    raise ValueError(f"[error] unknown lifelong.buffer_selection {mode}")
                np.save(path, chosen.astype(np.int64))
                print(f"[info] fixed buffer: task {k} share selected ({mode}), {q} of {n}")
            chosen = np.load(path)
            assert chosen.max() < n, f"[error] {path} does not fit task {k} ({n} samples)"
            parts.append(off + chosen)
            quotas.append(len(chosen))
        print(f"[info] fixed buffer, per task {quotas}")
        return np.concatenate(parts)

    def _rainbow_selection(self, past_data, sizes, offsets, quotas, tag=None):
        t0 = time.time()
        self.policy.eval()
        try:
            parts, log = [], {}
            for k, (off, n, q) in enumerate(zip(offsets, sizes, quotas)):
                scores = self._disagreement(Subset(past_data, list(range(off, off + n))))
                order = np.argsort(scores, kind="stable")
                # evenly spaced ranks: the centers of q equal bins of the sorted scores
                ranks = np.floor((np.arange(q) + 0.5) * n / q).astype(np.int64)
                chosen = order[ranks]
                parts.append(off + chosen)
                log[f"scores_task{k}"] = scores
                log[f"chosen_task{k}"] = chosen
                print(
                    f"[info] rainbow task {k}: {q} of {n}, disagreement median {np.median(scores):.4f} "
                    f"(chosen min {scores[chosen].min():.4f}, max {scores[chosen].max():.4f})"
                )
        finally:
            self.policy.train()
            torch.cuda.empty_cache()
        out_dir = os.path.join(self.experiment_dir, "rainbow")
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, f"{tag or f'task{self.current_task}'}.npz"), quotas=quotas, **log)
        self.replay_cost_selection_s = time.time() - t0
        print(f"[info] rainbow buffer, per task {quotas.tolist()}, selection {time.time() - t0:.0f} s")
        return np.concatenate(parts)

    def _kmeans_selection(self, past_data, sizes, offsets, quotas, rng, tag=None):
        from libero.lifelong.algos.anchor_er import selection

        t0 = time.time()
        cfg = self.cfg.lifelong
        keys = list(cfg.get("representation_keys") or
                    ["agentview_rgb", "eye_in_hand_rgb", "gripper_states", "joint_states"])
        n_clusters = int(cfg.get("kmeans_n_clusters", 10))
        generator = torch.Generator().manual_seed(self.cfg.seed + self.current_task)
        self.policy.eval()
        try:
            reprs = self._representations(past_data, keys)
        finally:
            self.policy.train()
        latents, _ = selection.normalize_representations(reprs, keys)
        del reprs
        parts, log = [], {}
        for k, (off, n, q) in enumerate(zip(offsets, sizes, quotas)):
            labels = selection.kmeans(latents[off:off + n], n_clusters=min(n_clusters, q),
                                      n_init=cfg.get("kmeans_n_init", 20), generator=generator).cpu().numpy()
            members = [np.nonzero(labels == c)[0] for c in range(labels.max() + 1)]
            alloc = equal_quotas([len(m) for m in members], q, rng)
            chosen = np.concatenate([rng.choice(m, size=a, replace=False) for m, a in zip(members, alloc) if a > 0])
            parts.append(off + chosen)
            log[f"labels_task{k}"] = labels
            log[f"chosen_task{k}"] = chosen
            print(f"[info] kmeans task {k}: {q} of {n}, cluster sizes {[len(m) for m in members]}, "
                  f"taken {alloc.tolist()}")
        torch.cuda.empty_cache()
        out_dir = os.path.join(self.experiment_dir, "kmeans")
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, f"{tag or f'task{self.current_task}'}.npz"), quotas=quotas, **log)
        print(f"[info] kmeans buffer, per task {quotas.tolist()}, selection {time.time() - t0:.0f} s")
        return np.concatenate(parts)

    @torch.no_grad()
    def _representations(self, dataset, keys):
        """obs-encoder features {key: (N, D)} of the given keys (as AnchorER._representations)."""
        reprs = {}
        loader = DataLoader(dataset, batch_size=self.cfg.lifelong.get("selection_batch_size", 128),
                            shuffle=False, num_workers=self.cfg.train.num_workers)
        for data in loader:
            data = self.map_tensor_to_device(data)
            for key, value in self.policy.compute_obs_representations(data).items():
                if key in keys:
                    reprs.setdefault(key, []).append(value.float())
        return {key: torch.cat(v, dim=0) for key, v in reprs.items()}

    @torch.no_grad()
    def _disagreement(self, dataset):
        """AnchorER's noise-denoise action disagreement (lifelong.rainbow_* settings) per sample."""
        cfg = self.cfg.lifelong
        metric = cfg.get("rainbow_disagreement_metric", "se3")
        scores = []
        loader = DataLoader(dataset, batch_size=cfg.get("selection_batch_size", 128), shuffle=False,
                            num_workers=self.cfg.train.num_workers)
        for data in loader:
            data = self.map_tensor_to_device(data)
            scores.append(self.policy.compute_action_disagreement(
                data,
                noise_level=cfg.get("rainbow_noise_level", 20),
                ood_metric=metric,
                num_inference_steps=cfg.get("rainbow_inference_steps", 100),
                action_scale=cfg.get("rainbow_action_scale") if metric == "se3" else None,
            ).cpu())
        return torch.cat(scores).numpy()

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
