import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from libero.lifelong.algos.base import Sequential
from libero.lifelong.algos.er import ER, merge_datas
from libero.lifelong.algos.offset_generator import (OffsetGenerator, ResMLP, SharedOffsetGenerator, flow_sample,
                                                    smooth_action_chunks, time_emb)
from libero.lifelong.algos.rand_er import EndlessRandomSampler


def kmeans_medoids(X, k, generator, iters=50, n_init=3):
    """k-means++ / Lloyd on the GPU; the sample nearest to each centroid (unique indices, CPU)."""
    best_inertia, best_C = math.inf, None
    n = X.shape[0]
    for _ in range(n_init):
        first = int(torch.randint(n, (1,), generator=generator))
        idx = [first]
        d2 = ((X - X[first]) ** 2).sum(1)
        for _ in range(1, k):
            p = (d2 / d2.sum()).double().cpu()
            i = int(torch.multinomial(p, 1, generator=generator))
            idx.append(i)
            d2 = torch.minimum(d2, ((X - X[i]) ** 2).sum(1))
        C = X[idx].clone()
        for _ in range(iters):
            lab = torch.cdist(X, C).argmin(1)
            sums = torch.zeros_like(C).index_add_(0, lab, X)
            cnt = torch.bincount(lab, minlength=k).unsqueeze(1)
            C = torch.where(cnt > 0, sums / cnt.clamp(min=1), C)
        inertia = torch.cdist(X, C).min(1).values.pow(2).sum().item()
        if inertia < best_inertia:
            best_inertia, best_C = inertia, C
    return torch.unique(torch.cdist(best_C, X).argmin(1)).cpu()


def ridge_to_identity(X, Y, alpha=1.0):
    """W minimizing ||X W - Y||^2 + lam ||W - I||^2, lam = alpha * mean eigenvalue of X^T X."""
    XtX = X.T @ X
    lam = alpha * torch.trace(XtX) / XtX.shape[0]
    eye = torch.eye(XtX.shape[0], device=X.device, dtype=X.dtype)
    return torch.linalg.solve(XtX + lam * eye, X.T @ Y + lam * eye)


def demo_and_time(dataset):
    """(demo number, timestep) of every sample of a SequenceVLDataset (robomimic index maps)."""
    sd = dataset.sequence_dataset
    demo_no = {d: k for k, d in enumerate(sd.demos)}
    demo, ts = [], []
    for i in range(len(sd)):
        d = sd._index_to_demo_id[i]
        demo.append(demo_no[d])
        ts.append(i - sd._demo_id_to_start_indices[d])
    return torch.tensor(demo), torch.tensor(ts)


def temporal_seeds(demo, ts, n_seeds, stagger=False):
    """
    n_seeds seeds spread over the demos in proportion to their length; one per equal contiguous segment of its demo,
    the middle frame, or with stagger=True at relative position (k + 0.5) / n_demos of the segment in demo k, so that
    across the demos the seeds cover all phases of the task evenly (at 1% each demo gets one seed: middle frames never
    show the start of a task, ss10_t1_er_rr2). Returns seed indices and the segment (seed) of every sample.
    """
    n_demo = int(demo.max()) + 1
    lens = torch.bincount(demo, minlength=n_demo).float()
    per = torch.clamp(torch.round(lens / lens.sum() * n_seeds), min=1).long()
    seeds, assign = [], torch.empty_like(demo)
    start = 0
    for k in range(n_demo):
        L, m = int(lens[k]), int(per[k])
        seg = torch.clamp((torch.arange(L) * m) // L, max=m - 1)
        phase = (k + 0.5) / n_demo if stagger else 0.5
        for q in range(m):
            idx = torch.nonzero(seg == q).squeeze(1)
            seeds.append(start + int(idx[min(int(phase * len(idx)), len(idx) - 1)]))
            assign[start + idx] = len(seeds) - 1
        start += L
    return torch.tensor(seeds), assign


def trajectory_seeds(demo, ts, n_seeds, spacing):
    """
    n_seeds seeds every `spacing` frames along whole demos (frames 0, spacing, 2 * spacing, ...), demo after demo, the
    demos taken evenly spread over the dataset until the budget is used: few demos, but each covered over its whole
    length, and neighbouring seeds `spacing` frames apart give exact old-task pairs for an autoregressive generator.
    Members: samples of covered demos less than `spacing` frames from a seed, assigned to the nearest one (at most
    spacing / 2 away, the tail after the last seed up to spacing - 1); all other samples get assign -1 (not replayed).
    """
    n_demo = int(demo.max()) + 1
    lens = torch.bincount(demo, minlength=n_demo)
    starts = torch.cumsum(lens, 0) - lens
    per_demo = (lens + spacing - 1) // spacing
    n_cover = max(1, round(n_seeds / per_demo.float().mean().item()))
    order = torch.linspace(0, n_demo - 1, min(n_cover, n_demo)).round().long().unique()
    seeds, assign = [], torch.full_like(demo, -1)
    for k in order.tolist():
        if len(seeds) >= n_seeds:
            break
        frames = torch.arange(0, int(lens[k]), spacing)[: n_seeds - len(seeds)]
        first = len(seeds)
        seeds += (int(starts[k]) + frames).tolist()
        t = torch.arange(int(lens[k]))
        dist, nearest = (t[:, None] - frames[None]).abs().min(1)
        near = dist < spacing  # a demo cut off by the budget keeps only the frames near its seeds
        assign[int(starts[k]) + t[near]] = first + nearest[near]
    return torch.tensor(seeds), assign


class Indexed(torch.utils.data.Dataset):
    """A dataset whose samples also carry their index (seed_index)."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        data = self.dataset[i]
        data["seed_index"] = torch.tensor(i, dtype=torch.long)
        return data


def step_actions(dataset):
    """Action of every sample's own time step (the first action of its chunk), (N, ac_dim), from the demos' arrays."""
    sd = dataset.sequence_dataset
    out = torch.cat([torch.as_tensor(sd.get_dataset_for_ep(d, "actions"), dtype=torch.float32) for d in sd.demos])
    assert len(out) == len(dataset), (len(out), len(dataset))
    return out


def action_paths(steps, i, j, length):
    """
    Actions from the earlier of the frames i and j up to the later one (exclusive, chronological), zero-padded or cut
    to length steps: (B, length, ac_dim) for index tensors i, j into steps (same demo).
    """
    lo, n = torch.minimum(i, j), (j - i).abs().clamp(max=length)
    k = torch.arange(length, device=i.device)
    idx = (lo[:, None] + k[None]).clamp(max=len(steps) - 1)
    return steps[idx] * (k[None] < n[:, None])[..., None]


class WithPartner(torch.utils.data.Dataset):
    """
    Current-task samples with a partner sample of the same demo (lifelong.latent_pred_weight): 1..max_dt steps later,
    or with signed=True up to max_dt steps before or after. data["partner"] = {"obs": its observations, "dt": signed
    steps}, with path_len > 0 also "path", the actions between the two frames (action_paths). dt 0 (partner = the
    sample) when no other step of the demo is in range.
    """

    def __init__(self, dataset, max_dt, signed=False, path_len=0):
        self.dataset, self.max_dt, self.signed, self.path_len = dataset, max_dt, signed, path_len
        demo, ts = demo_and_time(dataset)
        demo_len = torch.bincount(demo)
        self.steps_left = demo_len[demo] - 1 - ts
        self.steps_before = ts
        self.steps = step_actions(dataset) if path_len > 0 else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        data = self.dataset[i]
        hi = min(self.max_dt, int(self.steps_left[i]))
        lo = -min(self.max_dt, int(self.steps_before[i])) if self.signed else 1
        choices = [d for d in range(lo, hi + 1) if d != 0]
        dt = choices[int(torch.randint(len(choices), (1,)))] if choices else 0
        data["partner"] = {"obs": self.dataset[i + dt]["obs"], "dt": torch.tensor(dt, dtype=torch.long),
                           "index": torch.tensor(i, dtype=torch.long)}
        if self.path_len > 0:
            data["partner"]["path"] = action_paths(self.steps, torch.tensor([i]), torch.tensor([i + dt]), self.path_len)[0]
        return data


class WithNeighbor(torch.utils.data.Dataset):
    """Seed replay samples (seed_index) with the observations of a neighbouring seed of their demo (neighbor_obs)."""

    def __init__(self, seeds, neighbors):
        self.seeds, self.neighbors = seeds, neighbors

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, i):
        data = self.seeds[i]
        data["seed_index"] = torch.tensor(i, dtype=torch.long)
        data["neighbor_obs"] = self.neighbors[i]["obs"]
        return data


class WithSeedPartner(torch.utils.data.Dataset):
    """
    Seed replay samples (seed_index) with the observations of a random other seed of their demo (neighbor_obs,
    neighbor_index), drawn anew every time: all seed pairs of a demo, both directions (latent_pred_ar_len).
    """

    def __init__(self, seeds, partners):
        self.seeds, self.partners = seeds, partners  # partners[i]: indices into seeds, same demo

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, i):
        data = self.seeds[i]
        p = self.partners[i]
        k = p[int(torch.randint(len(p), (1,)))] if len(p) else i
        data["seed_index"] = torch.tensor(i, dtype=torch.long)
        data["neighbor_index"] = torch.tensor(k, dtype=torch.long)
        data["neighbor_obs"] = self.seeds[k]["obs"]
        return data


class LatentHead(nn.Module):
    """Predicts the image latent offset F(x) - F(s) of a partner x of s (lifelong.latent_pred_weight)."""

    def __init__(self, d_in, d_out, width, blocks=3):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                                   nn.Linear(width, width)) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d_out))
        # starts at offset 0: a random initial output, divided by the tiny latent variance of a fresh encoder, gave a
        # first-step loss of ~670 that would steer the encoder
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


class FlowOffsetHead(nn.Module):
    """
    Conditional rectified flow of the image latent offset F(x) - F(s) (lifelong.latent_pred_head=flow): ResMLP velocity
    field on [noisy offset, time embedding] given the condition; offsets divided by a running scale (std, momentum 0.99).
    Output layer zero-initialized (velocity 0 at the start).
    """

    def __init__(self, R, d_cond, width):
        super().__init__()
        self.net = ResMLP(R + 32, d_cond, R, width)
        nn.init.zeros_(self.net.out[1].weight)
        nn.init.zeros_(self.net.out[1].bias)
        self.register_buffer("scale", torch.tensor(0.0))

    def loss(self, c, y):
        with torch.no_grad():
            s = y.detach().std()
            self.scale.copy_(s if self.scale == 0 else 0.99 * self.scale + 0.01 * s)
        y = y / self.scale.clone()  # several loss calls per step update the buffer in place
        z = torch.randn_like(y)
        t = torch.rand(len(y), device=y.device)
        xt = (1 - t[:, None]) * z + t[:, None] * y
        v = self.net(torch.cat([xt, time_emb(t)], 1), c)
        return ((v - (y - z)) ** 2).mean()

    @torch.no_grad()
    def sample(self, c, n_steps, generator=None):
        return flow_sample(self, c, self.net.out[1].out_features, n_steps, generator) * self.scale

    def _cond(self, c):  # flow_sample calls model._cond
        return c


class SeedDriftER(ER):
    """
    Latent replay with seeded drift compensation (no pretraining, the obs encoder keeps training).

    At the end of task j: all of its training samples are encoded with the current encoder F_j (eval
    mode, center crop) and stored as latents (global condition) + action chunks. seed_fraction of them
    (k-means medoids on the image part of the latent) are kept raw as seeds; every other sample is
    assigned to its nearest seed s(x).

    While training task t, every batch = current-task batch + replay_ratio * batch_size raw seeds of all
    past tasks (through the encoder, with gradient, i.e. a fixed k-means raw buffer as RandER 1%) +
    latent_replay_ratio * batch_size stored latents of past non-seed samples (into the U-Net only).
    Every refit_every steps the seeds are re-encoded with the current encoder F_t (eval mode) and the
    replayed latents are rebuilt (only the image part; low-dim obs and task embedding do not drift):
      none             F_j(x)                                         (stale latents, control)
      seed_offset      F_t(s) + F_j(x) - F_j(s)
      seed_offset_map  F_t(s) + (F_j(x) - F_j(s)) W_j                 (default)
      seed_offset_grad F_t(s) + F_j(x) - F_j(s), with F_t(s) taken from the raw seed replay of the same
                       step (train mode, with encoder gradient): for every replayed seed one random member
                       x of its cluster is added as offset latent with action a(x), instead of the
                       separate latent batch (no extra encoder pass; refit_every unused)
      seed_only_grad   as seed_offset_grad with offset 0: F_t(s) paired with a(x) (no-perturbation control)
      noise_grad       as seed_offset_grad with a random-direction offset of the stored offset's norm
                       (unstructured-perturbation control)
      gen_grad         as seed_offset_grad with the offset sampled from the task's generator
                       (lifelong.generator cond | joint, algos/offset_generator.py); joint also generates the
                       action chunk and proprio of the replayed sample
    lifelong.seed_selection: kmeans (medoids of the image latents) | temporal (seeds spread over the demos in
    proportion to their length, each the middle frame of an equal contiguous segment of its demo, members =
    that segment; dt = (t - t_seed) / 50 is a generator input)
    W_j: ridge map (pulled to the identity) from the centered F_j(seeds of task j) to the centered
    F_t(seeds of task j), fit directly from j to the current encoder (never chained).
    Stored latents are saved to <run>/seed_drift/task<j>.pt and reloaded when a run is resumed (the
    encoder at resume time is no longer F_j).
    lifelong.latent_pred_weight > 0: auxiliary latent prediction on every task (task 0 included). Every current-task
    sample s gets a partner x of its demo 1..latent_pred_max_dt steps later; LatentHead predicts F(x) - F(s) from F(s),
    proprio of s and x, the action chunk of s up to x and dt (no task embedding), target F(x) without gradient, both
    encoded without augmentation; the loss reaches the encoder through F(s). Tests whether an encoder trained to make
    its latents predictable from the motion lets an offset predictor transfer to old tasks
    (experiments/2026-11-23_latent_predictor). The head is not used for replay.
    lifelong.latent_pred_head=flow: the head is a conditional flow (FlowOffsetHead) of the offset given F(s), proprio of s
    and x, the full action path between them and the signed steps (latent_pred_signed: partners before and after s).
    Extra flow targets on old tasks: latent_pred_seed_pairs (neighbouring raw seeds of a demo, both encoded now) and
    latent_pred_self_replay (offsets sampled from the head at the start of the task). Temporal seeds only.
      raw_grad         plain ER on the raw seeds only (no latent replay): an ER baseline on exactly the same buffer
      pred_grad        as seed_offset_grad with the offset sampled from the flow head in the current encoder's space,
                       F_t(s) + G(F_t(s), proprio, actions s -> x, dt): nothing stored per sample but proprio and the
                       action sequence
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.store = []  # per past task: dict(cond, actions, seeds, assign, dt)
        self.generators = []  # per past task: OffsetGenerator (lifelong.generator != none)
        self.shared_gen = None  # SharedOffsetGenerator of all past tasks (lifelong.gen_shared)
        self.seed_loader = None
        self.n_steps = 0
        self.replay_cost = {}
        self.latent_head = None
        self.aux_sum, self.aux_n = 0.0, 0
        self.aux_var = None
        self.target_encoder = None
        self.latent_std = None
        self.sr_cond = self.sr_y = None

    # ------------------------------------------------------------------ storage at the end of a task
    @torch.no_grad()
    def _encode(self, dataset):
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=self.cfg.train.num_workers)
        conds, acts = [], []
        was_training = self.policy.training
        self.policy.eval()
        for data in loader:
            c, a = self.policy.encode_latents(self.map_tensor_to_device(data))
            conds.append(c.float())
            acts.append(a.float())
        self.policy.train(was_training)
        return torch.cat(conds), torch.cat(acts)

    def _image_dim(self):
        if not hasattr(self, "R"):
            dims = self.policy.latent_dims()
            rgb = list(self.cfg.data.obs.modality.rgb)
            assert list(dims)[: len(rgb)] == rgb, dims
            self.R = sum(dims[k] for k in rgb)
        return self.R

    def learn_one_task(self, dataset, task_id, benchmark, result_summary):
        if self._ar_len() > 0:
            # proprio of every frame and the action of every step, for the intermediate frames of the chains
            R, P = self._image_dim(), self._prop_dim()
            self.cur_prop = self._encode(dataset)[0][:, R:R + P]
            self.cur_steps = step_actions(dataset).to(self.cfg.device)
        if self.cfg.lifelong.get("latent_pred_weight", 0.0) > 0:
            dataset = WithPartner(dataset, self.cfg.lifelong.latent_pred_max_dt,
                                  self.cfg.lifelong.get("latent_pred_signed", False), self._path_len() if self._flow() else 0)
        return super().learn_one_task(dataset, task_id, benchmark, result_summary)

    def end_task(self, dataset, task_id, benchmark, env=None):
        if isinstance(dataset, WithPartner):
            dataset = dataset.dataset
        if self.cfg.lifelong.get("latent_pred_weight", 0.0) > 0:
            self._save_or_load_latent_head(task_id)
        super().end_task(dataset, task_id, benchmark)  # self.datasets.append
        path = os.path.join(self.experiment_dir, "seed_drift", f"task{task_id}.pt")
        R = self._image_dim()
        if os.path.exists(path):
            entry = torch.load(path, map_location=self.cfg.device)
            print(f"[info] SeedDriftER: stored latents of task {task_id} loaded from {path}")
        else:
            t0 = time.time()
            cond, acts = self._encode(dataset)
            X = cond[:, :R]
            k = max(2, math.ceil(self.cfg.lifelong.seed_fraction * X.shape[0]))
            demo, ts = demo_and_time(dataset)
            seed_run = self.cfg.lifelong.get("seed_from_run", None)
            if seed_run:
                # the seed frames of another run (same benchmark, order, demos): same-buffer comparisons with
                # k-means seeds, which depend on the run's own encoder; members assigned to the nearest seed here
                seeds = torch.load(os.path.join(seed_run, "seed_drift", f"task{task_id}.pt"), map_location=X.device)["seeds"]
                assign = torch.cdist(X, X[seeds]).argmin(1)
            elif self.cfg.lifelong.get("seed_selection", "kmeans") == "temporal":
                seeds, assign = temporal_seeds(demo, ts, k, self.cfg.lifelong.get("seed_stagger", False))
                seeds, assign = seeds.to(X.device), assign.to(X.device)
            elif self.cfg.lifelong.seed_selection == "trajectory":
                seeds, assign = trajectory_seeds(demo, ts, k, self.cfg.lifelong.get("seed_spacing", 32))
                seeds, assign = seeds.to(X.device), assign.to(X.device)
            else:
                g = torch.Generator().manual_seed(self.cfg.seed * 1000 + task_id)
                seeds = kmeans_medoids(X, k, g, n_init=self.cfg.lifelong.kmeans_n_init).to(X.device)
                assign = torch.cdist(X, X[seeds]).argmin(1)  # index into seeds
            ts = ts.to(X.device)
            dt = (ts - ts[seeds][assign]).float() / 50.0
            entry = {"cond": cond, "actions": acts, "seeds": seeds, "assign": assign, "dt": dt}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(entry, path)
            print(f"[info] SeedDriftER: task {task_id}: {X.shape[0]} samples stored as latents, "
                  f"{len(seeds)} {self.cfg.lifelong.get('seed_selection', 'kmeans')} seeds raw ({len(seeds) / X.shape[0]:.2%}), {time.time() - t0:.0f} s")
        self.store.append(entry)
        if self.cfg.lifelong.get("generator", "none") != "none":
            if self.cfg.lifelong.get("gen_shared", False):
                self._fit_shared_generator(task_id)
            else:
                self.generators.append(self._fit_generator(entry, task_id))

    def _gen_inputs(self, z_seed, prop, act_flat, dt, temb=None):
        """Generator condition: seed latent F_j(s), (proprio, action if conditional), dt, (task embedding if shared)."""
        parts = [z_seed]
        if self.cfg.lifelong.generator == "cond":
            parts += [prop, act_flat]
        parts.append(dt[:, None])
        if temb is not None:
            parts.append(temb)
        return torch.cat(parts, 1)

    def _shared_gen_data(self, e):
        """Offsets and shared-generator conditions of an entry's non-seed samples."""
        R, P = self._image_dim(), self._prop_dim()
        ns = self._nonseed(e)
        img = e["cond"][:, :R]
        zs = img[e["seeds"]][e["assign"]]
        act = e["actions"].reshape(len(e["actions"]), -1)
        C = self._gen_inputs(zs, e["cond"][:, R:R + P], act, e["dt"], e["cond"][:, R + P:])
        return (img - zs)[ns], C[ns]

    def _fit_shared_generator(self, task_id):
        """Stage task_id of the continual shared generator (one network, per-task PCA bases, self-replay)."""
        lc = self.cfg.lifelong
        R = self._image_dim()
        path = os.path.join(self.experiment_dir, "seed_drift", f"gen_shared_task{task_id}.pt")
        d, C = self._shared_gen_data(self.store[task_id])
        dev = d.device
        if self.shared_gen is None:
            self.shared_gen = SharedOffsetGenerator(R, C.shape[1] - R, lc.gen_pca, lc.gen_width).to(dev)
        if os.path.exists(path):
            self.shared_gen.ensure_bases(task_id + 1, dev)
            self.shared_gen.load_state_dict(torch.load(path, map_location=dev))
            self.shared_gen.eval()
            print(f"[info] SeedDriftER: shared generator after task {task_id} loaded from {path}")
        else:
            t0 = time.time()
            old_C = torch.cat([self._shared_gen_data(e)[1] for e in self.store[:task_id]]) if task_id > 0 else None
            for q in self.shared_gen.parameters():  # frozen after the previous stage
                q.requires_grad_(True)
            loss = self.shared_gen.fit_task(d, C, old_C, lc.gen_train_steps, lc.get("gen_new_task_share", 0.5),
                                            lc.gen_sample_steps, seed=self.cfg.seed * 1000 + task_id)
            torch.save(self.shared_gen.state_dict(), path)
            print(f"[info] SeedDriftER: shared generator stage {task_id} ({len(d)} real + "
                  f"{0 if old_C is None else len(old_C)} self-replayed samples): "
                  f"{self.shared_gen.storage_bytes() / 1e6:.1f} MB fp16, final loss {loss:.3f}, {time.time() - t0:.0f} s")
        for q in self.shared_gen.parameters():
            q.requires_grad_(False)

    def _fit_generator(self, e, task_id):
        R, P = self._image_dim(), self._prop_dim()
        lc = self.cfg.lifelong
        path = os.path.join(self.experiment_dir, "seed_drift", f"gen_task{task_id}.pt")
        nonseed = self._nonseed(e)
        img = e["cond"][:, :R]
        zs = img[e["seeds"]][e["assign"]]
        prop, act = e["cond"][:, R:R + P], e["actions"].reshape(len(e["actions"]), -1)
        C = self._gen_inputs(zs, prop, act, e["dt"])[nonseed]
        gen = OffsetGenerator(R, C.shape[1] - R, lc.gen_pca, act.shape[1], P, lc.generator == "joint",
                              lc.gen_width).to(img.device)
        if os.path.exists(path):
            gen.load_state_dict(torch.load(path, map_location=img.device))
            gen.eval()
            print(f"[info] SeedDriftER: generator of task {task_id} loaded from {path}")
        else:
            t0 = time.time()
            loss = gen.fit((img - zs)[nonseed], act[nonseed], prop[nonseed], C, lc.gen_train_steps,
                           seed=self.cfg.seed * 1000 + task_id)
            torch.save(gen.state_dict(), path)
            print(f"[info] SeedDriftER: generator ({lc.generator}) of task {task_id}: "
                  f"{gen.storage_bytes() / 1e6:.1f} MB fp16, final loss {loss:.3f}, {time.time() - t0:.0f} s")
        for q in gen.parameters():
            q.requires_grad_(False)
        return gen

    def _prop_dim(self):
        dims = self.policy.latent_dims()
        return sum(d for k, d in dims.items() if k not in self.cfg.data.obs.modality.rgb and k != "task_emb")

    # ------------------------------------------------------------------ replay setup
    def start_task(self, task):
        Sequential.start_task(self, task)  # not ER.start_task (truncated per-task buffers)
        if self.cfg.lifelong.get("latent_pred_weight", 0.0) > 0:
            self.optimizer.add_param_group({"params": self._get_latent_head().parameters()})
        self.seed_loader = None
        if self.current_task <= 0:
            return
        assert len(self.store) == len(self.datasets) == task, "[error] SeedDriftER store out of sync"
        dev = self.cfg.device
        bs = self.cfg.train.batch_size
        # raw seeds of all past tasks: replay loader (encoder gradient) and a cached copy for refits
        seed_sets = [Subset(ds, e["seeds"].cpu().tolist()) for ds, e in zip(self.datasets, self.store)]
        seed_all = torch.utils.data.ConcatDataset(seed_sets)
        lc = self.cfg.lifelong
        uses_paths = self._flow() and (lc.drift_compensation == "pred_grad" or lc.get("latent_pred_self_replay", False)
                                        or lc.get("latent_pred_seed_pairs", False))
        if uses_paths:
            assert lc.get("seed_selection", "kmeans") in ("temporal", "trajectory"), \
                "[error] action paths need temporal or trajectory seeds (same demo)"
        seed_ds = Indexed(seed_all)
        if self._ar_len() > 0:
            self._setup_ar_tables()
        if self._flow() and lc.get("latent_pred_seed_pairs", False):
            if self._ar_len() > 0:
                seed_ds = WithSeedPartner(seed_all, self._seed_partners())
            else:
                seed_ds = WithNeighbor(seed_all, torch.utils.data.ConcatDataset(self._seed_neighbors()))
        self.seed_loader = iter(DataLoader(
            seed_ds, batch_size=max(1, round(self.cfg.lifelong.replay_ratio * bs)),
            num_workers=self.cfg.train.num_workers, sampler=EndlessRandomSampler(len(seed_all), self.cfg.seed + task)))
        self.seed_batches = [self.map_tensor_to_device(d) for d in
                             DataLoader(seed_all, batch_size=128, shuffle=False, num_workers=self.cfg.train.num_workers)]

        # replayed latents: the non-seed samples of all past tasks
        R = self._image_dim()
        off, low, acts, task_of, seed_of, sample_of = [], [], [], [], [], []
        seed_base = 0
        self.seed_task, self.seed_stale = [], []
        for j, e in enumerate(self.store):
            nonseed = self._nonseed(e).to(dev)
            img = e["cond"][:, :R]
            s_img = img[e["seeds"]]
            off.append((img - s_img[e["assign"]])[nonseed])
            low.append(e["cond"][nonseed, R:])
            acts.append(e["actions"][nonseed])
            task_of.append(torch.full((int(nonseed.sum()),), j, device=dev))
            seed_of.append(e["assign"][nonseed] + seed_base)
            sample_of.append(torch.nonzero(nonseed).squeeze(1))
            self.seed_task.append(torch.full((len(e["seeds"]),), j, device=dev))
            self.seed_stale.append(s_img)
            seed_base += len(e["seeds"])
        self.lat_off, self.lat_low = torch.cat(off), torch.cat(low)
        self.lat_actions, self.lat_task, self.lat_seed = torch.cat(acts), torch.cat(task_of), torch.cat(seed_of)
        self.lat_dt = torch.cat([e["dt"][self._nonseed(e)] for e in self.store])
        self.lat_sample = torch.cat(sample_of)  # index of every replayed latent in its task's dataset
        self.seed_task, self.seed_stale = torch.cat(self.seed_task), torch.cat(self.seed_stale)
        self.seed_var = self.seed_stale.var(0).mean().clamp(min=1e-8)  # scale of the seed distillation loss
        self.lat_gen = torch.Generator(device=dev).manual_seed(self.cfg.seed + task)
        # non-seed members of every seed's cluster (seed_offset_grad draws one per replayed seed)
        self.member_order = torch.argsort(self.lat_seed)
        self.seed_count = torch.bincount(self.lat_seed, minlength=len(self.seed_stale))
        self.seed_start = torch.cumsum(self.seed_count, 0) - self.seed_count
        self.lat_perm, self.lat_pos = None, 0
        self.lat_bs = max(1, round(self.cfg.lifelong.latent_replay_ratio * bs))
        self.n_steps = 0
        self.sr_cond = self.sr_y = None
        if uses_paths:
            self._setup_flow_paths()
            if lc.get("latent_pred_self_replay", False):
                self._setup_self_replay()
        self._refit()

        n_seed, n_lat = len(self.seed_stale), len(self.lat_off)
        raw_b, lat_b = self._bytes_per_sample()
        self.replay_cost = {
            "replay": f"seed_drift ({self.cfg.lifelong.drift_compensation})",
            "n_seed_raw": n_seed, "n_latent": n_lat,
            "raw_bytes_per_sample": raw_b, "latent_bytes_per_sample": lat_b,
            "buffer_bytes": int(n_seed * raw_b + (n_seed + n_lat) * lat_b),
            "buffer_bytes_if_all_raw": int((n_seed + n_lat) * raw_b),
        }
        if self.generators or self.shared_gen is not None:
            gen_b = self.shared_gen.storage_bytes() if self.shared_gen is not None else sum(g.storage_bytes() for g in self.generators)
            side = 0 if self.cfg.lifelong.generator == "joint" else 2 * (self._prop_dim() + self.lat_actions[0].numel())
            self.replay_cost.update({
                "generator_bytes_fp16": int(gen_b), "side_bytes_per_sample_fp16": int(side),
                "buffer_bytes_generator": int(n_seed * raw_b + n_seed * 2 * R + n_lat * (side + 2) + gen_b)})
            print(f"[info] SeedDriftER generator storage: {gen_b / 1e6:.1f} MB generators + {side} B/sample "
                  f"side info, total {self.replay_cost['buffer_bytes_generator'] / 1e6:.1f} MB")
        n_save = self.cfg.lifelong.get("gen_save_samples", 0)
        if n_save > 0 and (self.generators or self.shared_gen is not None):
            self._save_generated_samples(task, n_save)
        print(f"[info] SeedDriftER: {n_seed} raw seeds + {n_lat} latents of {task} past tasks, "
              f"compensation {self.cfg.lifelong.drift_compensation}, "
              f"{self.replay_cost['buffer_bytes'] / 1e6:.1f} MB (all raw: "
              f"{self.replay_cost['buffer_bytes_if_all_raw'] / 1e6:.1f} MB)")

    @staticmethod
    def _nonseed(e):
        """Replayed latents: samples that are not seeds and have a seed (trajectory seeds leave demos uncovered)."""
        m = e["assign"] >= 0
        m[e["seeds"]] = False
        return m

    def _bytes_per_sample(self):
        meta = self.cfg.shape_meta["all_shapes"]
        n_obs = self.cfg.policy.n_obs_steps
        action_bytes = 4 * self.cfg.policy.horizon * self.cfg.shape_meta["ac_dim"]
        raw = action_bytes + n_obs * (
            sum(int(np.prod(meta[k])) for k in self.cfg.data.obs.modality.rgb)
            + 4 * sum(int(np.prod(meta[k])) for k in self.cfg.data.obs.modality.low_dim))
        dims = self.policy.latent_dims()
        lat = action_bytes + 4 * sum(d for k, d in dims.items() if k != "task_emb")
        return int(raw), int(lat)

    # ------------------------------------------------------------------ drift compensation
    @torch.no_grad()
    def _refit(self):
        mode = self.cfg.lifelong.drift_compensation
        R = self._image_dim()
        if mode == "none":
            self.lat_img = self.seed_stale[self.lat_seed] + self.lat_off
            return
        if mode.endswith("_grad"):
            self.lat_img = None  # built per step in observe
            return
        was_training = self.policy.training
        self.policy.eval()
        cur = torch.cat([self.policy.encode_latents(d)[0].float()[:, :R] for d in self.seed_batches])
        self.policy.train(was_training)
        base = cur[self.lat_seed]
        if mode == "seed_offset":
            self.lat_img = base + self.lat_off
        elif mode == "seed_offset_map":
            img = torch.empty_like(self.lat_off)
            alpha = self.cfg.lifelong.ridge_alpha
            drift = []
            for j in range(len(self.store)):
                sm = self.seed_task == j
                sj, st = self.seed_stale[sm], cur[sm]
                W = ridge_to_identity(sj - sj.mean(0), st - st.mean(0), alpha)
                lm = self.lat_task == j
                img[lm] = base[lm] + self.lat_off[lm] @ W
                drift.append(((st - sj) ** 2).sum(1).mean().item() / ((st - st.mean(0)) ** 2).sum(1).mean().item())
            self.lat_img = img
            if self.n_steps == 0:
                print("[info] SeedDriftER seed drift (err/var of stale seed latents) per past task: "
                      + " ".join(f"{d:.2f}" for d in drift))
        else:
            raise ValueError(f"[error] unknown lifelong.drift_compensation {mode}")

    def _next_latent_indices(self):
        n, out, need = len(self.lat_off), [], self.lat_bs
        while need > 0:
            if self.lat_perm is None or self.lat_pos >= n:
                self.lat_perm = torch.randperm(n, generator=self.lat_gen, device=self.lat_off.device)
                self.lat_pos = 0
            take = self.lat_perm[self.lat_pos: self.lat_pos + need]
            self.lat_pos += len(take)
            need -= len(take)
            out.append(take)
        return torch.cat(out)

    def _step_offsets(self, mode, mem):
        """Offsets (F_j image space) added to the fresh seed latents for the cluster members mem."""
        if mode in ("seed_offset_grad", "raw_grad"):  # raw_grad: mem is empty
            return self.lat_off[mem]
        if mode == "seed_only_grad":
            return torch.zeros_like(self.lat_off[mem])
        if mode == "noise_grad":
            d = self.lat_off[mem]
            e = torch.randn(d.shape, device=d.device, generator=self.lat_gen)
            return e / e.norm(dim=1, keepdim=True) * d.norm(dim=1, keepdim=True)
        raise ValueError(f"[error] unknown lifelong.drift_compensation {mode}")

    def _seed_distill_loss(self, nobs, n_cur, sidx):
        """
        lifelong.seed_distill_weight: the raw seeds of this step through the encoder again without augmentation (eval
        transforms, as their stored latents; with gradient), pulled toward their latents F_j(s) stored at the end of
        their task, relative to the stored seeds' mean per-dimension variance. Less encoder drift where the old tasks
        live keeps their stored offsets valid (experiments/2026-11-23_spatial_drift).
        """
        enc = self.policy.dp.obs_encoder
        enc.eval()
        feat = enc({k: v[n_cur:] for k, v in nobs.items()})[:, : self._image_dim()]
        enc.train()
        return ((feat - self.seed_stale[sidx]) ** 2).mean() / self.seed_var

    @torch.no_grad()
    def _save_generated_samples(self, task, n):
        """
        For analysis (lifelong.gen_save_samples): per past task j, n random replayed (non-seed) samples with the
        generated and the true offset (F_j space), and the latent they are replayed as, F_t(s) + d_gen, next to the
        true current latent F_t(x) (image part, encoder at the start of task t, eval mode). Own RNG, so the
        training's random stream is unchanged. -> seed_drift/generated_task<t>.pt
        """
        R = self._image_dim()
        g = torch.Generator(device=self.lat_off.device).manual_seed(self.cfg.seed * 1000 + 777 + task)
        out = {}
        for j, e in enumerate(self.store):
            idx = torch.nonzero(self.lat_task == j).squeeze(1)
            mem = idx[torch.randperm(len(idx), generator=g, device=idx.device)[:n]]
            sample = self.lat_sample[mem]
            seed_sample = e["seeds"][e["assign"][sample]]
            d_gen = self._generate(mem, generator=g)[0]
            z_x = self._encode(Subset(self.datasets[j], sample.cpu().tolist()))[0][:, :R]
            z_s = self._encode(Subset(self.datasets[j], seed_sample.cpu().tolist()))[0][:, :R]
            out[j] = {"sample": sample.cpu(), "seed_sample": seed_sample.cpu(), "d_gen": d_gen.cpu(),
                      "d_true": self.lat_off[mem].cpu(), "z_seed_t": z_s.cpu(), "z_true_t": z_x.cpu(),
                      "actions": self.lat_actions[mem].cpu()}
        path = os.path.join(self.experiment_dir, "seed_drift", f"generated_task{task}.pt")
        torch.save(out, path)
        print(f"[info] SeedDriftER: {n} generated replay samples per past task saved to {path}")

    @torch.no_grad()
    def _generate(self, mem, generator=None):
        """Generated offsets, flat actions and proprio (the latter two None for conditional generators)."""
        generator = self.lat_gen if generator is None else generator
        R, P = self._image_dim(), self._prop_dim()
        joint = self.cfg.lifelong.generator == "joint"
        off = torch.empty_like(self.lat_off[mem])
        act = torch.empty(len(mem), self.lat_actions[0].numel(), device=off.device) if joint else None
        prop = torch.empty(len(mem), P, device=off.device) if joint else None
        task = self.lat_task[mem]
        for j in torch.unique(task).tolist():
            m = task == j
            mm = mem[m]
            shared = self.shared_gen is not None
            C = self._gen_inputs(self.seed_stale[self.lat_seed[mm]], self.lat_low[mm][:, :P],
                                 self.lat_actions[mm].reshape(len(mm), -1), self.lat_dt[mm],
                                 self.lat_low[mm][:, P:] if shared else None)
            if shared:
                d, a, p = self.shared_gen.sample(C, j, self.cfg.lifelong.gen_sample_steps, generator), None, None
            else:
                d, a, p = self.generators[j].sample(C, self.cfg.lifelong.gen_sample_steps, generator)
            off[m] = d
            if joint:
                act[m], prop[m] = a, p
        return off, act, prop

    # ------------------------------------------------------------------ auxiliary latent prediction
    def _flow(self):
        return self.cfg.lifelong.get("latent_pred_head", "regression") == "flow"

    def _path_len(self):
        return self.cfg.lifelong.get("latent_pred_path_len", 48)

    def _ar_len(self):
        L = self.cfg.lifelong.get("latent_pred_ar_len", 0)
        assert L == 0 or (self._flow() and L == self._path_len()), "[error] latent_pred_ar_len needs the flow head, path_len = ar_len"
        return L

    def _setup_ar_tables(self):
        """Proprio and step actions of all frames of all past tasks in one index space, every seed's frame in it."""
        R, P, dev = self._image_dim(), self._prop_dim(), self.cfg.device
        sizes = torch.tensor([len(e["cond"]) for e in self.store], device=dev)
        self.old_base = torch.cumsum(sizes, 0) - sizes
        self.old_prop = torch.cat([e["cond"][:, R:R + P] for e in self.store])
        self.old_steps = torch.cat([e["actions"][:, 0] for e in self.store]).to(dev)
        self.seed_g = torch.cat([b + e["seeds"].to(dev) for b, e in zip(self.old_base, self.store)])

    def _seed_partners(self):
        """For every seed (index into all past seeds) the other seeds of its demo."""
        out, base = [], 0
        for ds, e in zip(self.datasets, self.store):
            demo, _ = demo_and_time(ds)
            d = demo[e["seeds"].cpu()]
            for k in range(len(d)):
                out.append([base + int(m) for m in torch.nonzero(d == d[k]).squeeze(1) if int(m) != k])
            base += len(d)
        n_pairs = sum(len(p) for p in out)
        print(f"[info] SeedDriftER AR generator: {n_pairs} ordered seed pairs within demos "
              f"({sum(1 for p in out if p)} of {len(out)} seeds have a partner)")
        return out

    @torch.no_grad()
    def _ar_chain(self, z, a, b, prop, steps, to_last_step):
        """
        Steps of at most latent_pred_ar_len frames from frame a toward frame b (index tensors into prop / steps), each
        from the head's own sample of the previous one, starting at latent z. to_last_step: stop one step short (the
        caller trains that step, pushforward); else run to b. Returns the latent and the frame reached.
        """
        L, n_steps = self._ar_len(), self.cfg.lifelong.get("latent_pred_sample_steps", 10)
        z, cur = z.clone(), a.clone()
        while True:
            rem = b - cur
            i = torch.nonzero(rem.abs() > L if to_last_step else rem != 0).squeeze(1)
            if len(i) == 0:
                return z, cur
            nxt = cur[i] + rem[i].clamp(-L, L)
            c = self._flow_cond(z[i], prop[cur[i]], prop[nxt], action_paths(steps, cur[i], nxt, L), nxt - cur[i])
            z[i] = z[i] + self.latent_head.sample(c, n_steps)
            cur[i] = nxt

    def _ar_last_step(self, z, a, b, prop, steps):
        """Condition of the last step of the chain a -> b (single steps: from z itself, with its gradient)."""
        z_in, cur = self._ar_chain(z.detach(), a, b, prop, steps, to_last_step=True)
        z_in = torch.where((cur == a)[:, None], z, z_in)
        L = self._ar_len()
        return self._flow_cond(z_in, prop[cur], prop[b], action_paths(steps, cur, b, L), b - cur), z_in

    def _get_latent_head(self):
        if self.latent_head is None:
            R, P = self._image_dim(), self._prop_dim()
            width = self.cfg.lifelong.latent_pred_width
            if self._flow():
                d_cond = R + 2 * P + self._path_len() * self.cfg.shape_meta.ac_dim + 1
                self.latent_head = FlowOffsetHead(R, d_cond, width).to(self.cfg.device)
            else:
                n_act = self.policy.horizon * self.cfg.shape_meta.ac_dim
                self.latent_head = LatentHead(R + 2 * P + n_act + 1, R, width).to(self.cfg.device)
        return self.latent_head

    def _flow_cond(self, z_s, prop_s, prop_x, path, dt):
        """Flow head condition: seed latent, proprio of seed and sample, normalized action path, signed steps / path_len."""
        L = self._path_len()
        mask = (torch.arange(L, device=path.device)[None] < dt.abs().clamp(max=L)[:, None])[..., None]
        path = self.policy.dp.normalizer["action"].normalize(path) * mask
        return torch.cat([z_s, prop_s, prop_x, path.flatten(1), dt.float()[:, None] / L], 1)

    def _seed_neighbors(self):
        """
        lifelong.latent_pred_seed_pairs: for every seed of every past task the next (else the previous) seed of its demo
        at most latent_pred_path_len steps away, with the action path and signed steps to it. Returns per task a Subset
        of the neighbours (the seed itself where there is none; masked by seed_nb_valid).
        """
        L, dev = self._path_len(), self.cfg.device
        sets, valid, paths, dsteps = [], [], [], []
        for ds, e in zip(self.datasets, self.store):
            demo, _ = demo_and_time(ds)
            s = e["seeds"].cpu()
            d = demo[s]
            nxt, prv = torch.cat([s[1:], s[-1:]]), torch.cat([s[:1], s[:-1]])
            same_n = torch.cat([d[1:] == d[:-1], torch.tensor([False])]) & (nxt - s <= L)
            same_p = torch.cat([torch.tensor([False]), d[1:] == d[:-1]]) & (s - prv <= L)
            nb = torch.where(same_n, nxt, torch.where(same_p, prv, s))
            paths.append(action_paths(e["actions"][:, 0].to(dev), s.to(dev), nb.to(dev), L))
            dsteps.append((nb - s).to(dev))
            valid.append((same_n | same_p).to(dev))
            sets.append(Subset(ds, nb.tolist()))
        self.seed_nb_valid, self.seed_nb_path, self.seed_nb_dsteps = torch.cat(valid), torch.cat(paths), torch.cat(dsteps)
        print(f"[info] SeedDriftER flow head: {int(self.seed_nb_valid.sum())} of {len(self.seed_nb_valid)} seeds have a "
              f"neighbouring seed within {L} steps (mean |dt| {self.seed_nb_dsteps[self.seed_nb_valid].abs().float().mean():.1f})")
        return sets

    def _setup_flow_paths(self):
        """Action paths and signed steps from every replayed latent's seed to it (temporal seeds: same demo)."""
        L, dev = self._path_len(), self.cfg.device
        paths, dsteps = [], []
        for e in self.store:
            ns = torch.nonzero(self._nonseed(e)).squeeze(1)
            s = e["seeds"][e["assign"][ns]]
            paths.append(action_paths(e["actions"][:, 0].to(dev), s.to(dev), ns.to(dev), L))
            dsteps.append((ns - s).to(dev))
        self.lat_path, self.lat_dsteps = torch.cat(paths), torch.cat(dsteps)
        print(f"[info] SeedDriftER flow head: action paths of {len(self.lat_dsteps)} replayed latents, mean |dt| "
              f"{self.lat_dsteps.abs().float().mean():.1f} steps, {int((self.lat_dsteps.abs() > L).sum())} longer than {L} (cut)")

    @torch.no_grad()
    def _setup_self_replay(self):
        """
        lifelong.latent_pred_self_replay: offsets of all replayed latents sampled from the head as it is at the start of
        the task, with the seed latents of the encoder at that time; extra flow targets during the task, so the head keeps
        the old tasks' dynamics (each offset at most one task old).
        """
        R, P = self._image_dim(), self._prop_dim()
        was_training = self.policy.training
        self.policy.eval()
        cur = torch.cat([self.policy.encode_latents(d)[0].float() for d in self.seed_batches])
        self.policy.train(was_training)
        c = self._flow_cond(cur[self.lat_seed, :R], cur[self.lat_seed, R:R + P], self.lat_low[:, :P], self.lat_path,
                            self.lat_dsteps)
        if self._ar_len() > 0:  # single steps only: members at most ar_len frames from their seed
            c = c[self.lat_dsteps.abs() <= self._ar_len()]
        n_steps = self.cfg.lifelong.get("latent_pred_sample_steps", 10)
        self.sr_y = torch.cat([self._get_latent_head().sample(c[i: i + 4096], n_steps, self.lat_gen)
                               for i in range(0, len(c), 4096)])
        self.sr_cond = c
        print(f"[info] SeedDriftER flow head: {len(c)} self-replay targets sampled")

    def _old_flow_loss(self, data, cond, n_cur, sidx, nb_obs, nb_index=None):
        """Flow loss on old tasks: neighbouring seed pairs encoded now (seed_pairs) and self-replay targets."""
        R, P = self._image_dim(), self._prop_dim()
        terms = []
        if nb_obs is not None and self._ar_len() > 0:
            # a random other seed of the demo, chained (both encoded now: seed online, partner with the EMA target)
            obs = self.policy._dp_batch({"obs": self.map_tensor_to_device(nb_obs), "task_emb": data["task_emb"][n_cur:],
                                         "actions": data["actions"][n_cur:]})["obs"]
            with torch.no_grad():
                zn = self._get_target_encoder()(self.policy.dp.normalizer.normalize(obs))
            a, b = self.seed_g[sidx], self.seed_g[nb_index]
            v = a != b
            if v.any():
                c, z_in = self._ar_last_step(cond[n_cur:, :R][v], a[v], b[v], self.old_prop, self.old_steps)
                terms.append(self.latent_head.loss(c, zn[v, :R] - z_in.detach()))
        elif nb_obs is not None:
            v = self.seed_nb_valid[sidx]
            if v.any():
                obs = self.policy._dp_batch({"obs": self.map_tensor_to_device(nb_obs), "task_emb": data["task_emb"][n_cur:],
                                             "actions": data["actions"][n_cur:]})["obs"]
                with torch.no_grad():
                    zn = self._get_target_encoder()(self.policy.dp.normalizer.normalize(obs))[v]
                zs = cond[n_cur:][v]
                c = self._flow_cond(zs[:, :R], zs[:, R:R + P], zn[:, R:R + P], self.seed_nb_path[sidx[v]],
                                    self.seed_nb_dsteps[sidx[v]])
                terms.append(self.latent_head.loss(c, zn[:, :R] - zs[:, :R].detach()))
        if self.sr_cond is not None:
            i = torch.randint(len(self.sr_cond), (len(sidx),), device=self.sr_cond.device, generator=self.lat_gen)
            terms.append(self.latent_head.loss(self.sr_cond[i], self.sr_y[i]))
        return sum(terms) if terms else torch.zeros((), device=cond.device)

    def _save_or_load_latent_head(self, task_id):
        """Saved at the end of every task; a resumed run reloads the head of its last finished task."""
        path = os.path.join(self.experiment_dir, "seed_drift", f"latent_head_task{task_id}.pt")
        if os.path.exists(path):
            self._get_latent_head().load_state_dict(torch.load(path, map_location=self.cfg.device))
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self._get_latent_head().state_dict(), path)

    def _get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.policy.dp.obs_encoder).eval()
            for q in self.target_encoder.parameters():
                q.requires_grad = False
        return self.target_encoder

    @torch.no_grad()
    def _update_target_encoder(self):
        m = self.cfg.lifelong.latent_pred_target_momentum
        for t, o in zip(self.target_encoder.parameters(), self.policy.dp.obs_encoder.parameters()):
            t.mul_(m).add_(o.detach(), alpha=1 - m)

    def _latent_pred_loss(self, cur, partner):
        """
        Prediction error of F(x) from F(s) + head(...), relative to a running mean (momentum 0.99) of the batches' mean
        per-dimension variance of F(x). The target F(x) comes from an EMA copy of the encoder (momentum
        latent_pred_target_momentum, as BYOL): with the online encoder as target (705ebdc, runs
        ss_offset_grad_spatial5_lp01 / _lp10) the image latents collapsed (std ~1e-4 instead of ~0.15), and dividing by
        the batch's own variance gave loss spikes up to ~2500. The EMA target alone still shrank the latents (std 0.003
        vs 0.021 without the loss after 500 steps): the loss rewards removing what the head cannot predict from the
        motion. latent_pred_std_floor > 0 adds a VICReg variance term, mean_j relu(1 - std_j(F(s)) / floor) over the
        image dims of the batch.
        """
        policy, enc = self.policy, self.policy.dp.obs_encoder
        if cur["actions"].shape[0] < 2:
            # the batch std of a single sample is NaN (n - 1 = 0): the last batch of an epoch can have one sample
            # (ss10_t1_gen_sr went NaN in task 5 this way)
            return torch.zeros((), device=cur["actions"].device)
        R, P = self._image_dim(), self._prop_dim()
        s_batch = policy._dp_batch(cur)
        x_obs = policy._dp_batch({"obs": partner["obs"], "task_emb": cur["task_emb"], "actions": cur["actions"]})["obs"]
        enc.eval()
        zs = enc(policy.dp.normalizer.normalize(s_batch["obs"]))
        enc.train()
        with torch.no_grad():
            zx = self._get_target_encoder()(policy.dp.normalizer.normalize(x_obs))
        self.latent_std = zs[:, :R].detach().std(0).mean()
        if self._flow():
            # the encoder learns through the condition only: with F(s) in the target too it collapsed (smoke_flowgen:
            # latent std 0.002), since all-zero offsets make the flow trivial
            if self._ar_len() > 0:
                # chains of steps <= ar_len frames toward partners up to latent_pred_max_dt away (pushforward: only the
                # last step is trained, from the head's own rollout; single steps keep the encoder gradient)
                a = partner["index"]
                c, z_in = self._ar_last_step(zs[:, :R], a, a + partner["dt"], self.cur_prop, self.cur_steps)
                loss = self.latent_head.loss(c, zx[:, :R] - z_in.detach())
            else:
                c = self._flow_cond(zs[:, :R], zs[:, R:R + P], zx[:, R:R + P], partner["path"], partner["dt"])
                loss = self.latent_head.loss(c, zx[:, :R] - zs[:, :R].detach())
        else:
            dt = partner["dt"].float()
            act = policy.dp.normalizer["action"].normalize(s_batch["action"])
            act = act * (torch.arange(act.shape[1], device=act.device)[None, :, None] < dt[:, None, None])
            inp = torch.cat([zs[:, :R], zs[:, R:R + P], zx[:, R:R + P], act.flatten(1),
                             dt[:, None] / self.cfg.lifelong.latent_pred_max_dt], 1)
            pred = zs[:, :R] + self.latent_head(inp)
            var = zx[:, :R].var(0).mean()
            self.aux_var = var if self.aux_var is None else 0.99 * self.aux_var + 0.01 * var
            loss = ((pred - zx[:, :R]) ** 2).mean() / self.aux_var
        if self.cfg.lifelong.get("latent_pred_variance_only", False):
            loss = 0.0 * loss  # control: only the variance term below, same compute
        floor = self.cfg.lifelong.get("latent_pred_std_floor", 0.0)
        if floor > 0:
            loss = loss + torch.relu(1 - zs[:, :R].std(0) / floor).mean()
        return loss

    def _add_latent_pred_loss(self, loss, cur, partner):
        aux = self._latent_pred_loss(self.map_tensor_to_device(cur), self.map_tensor_to_device(partner))
        if self.n_steps == 1:
            print(f"[info] SeedDriftER: latent prediction loss at the first step of the task {aux.item():.4f} "
                  f"(policy loss {loss.item():.4f})")
        self.aux_sum, self.aux_n = self.aux_sum + aux.item(), self.aux_n + 1
        if self.aux_n == 500:
            print(f"[info] SeedDriftER: latent prediction loss, mean of the last 500 steps {self.aux_sum / 500:.4f}, "
                  f"image latent std {self.latent_std.item():.4f} (collapse check; ~0.13-0.16 without the loss)")
            self.aux_sum, self.aux_n = 0.0, 0
        if os.environ.get("SDR_NAN_DEBUG"):
            getattr(self, "dbg_parts", {}).update({"latent_pred_loss": aux.item()})
        return loss + self.cfg.lifelong.latent_pred_weight * aux

    def _nan_debug(self, loss):
        """SDR_NAN_DEBUG=1: every 50 steps (and at the first non-finite value, then stop) the loss, the gradient norms
        of policy and latent head (before clipping), the largest generated offset and seed latent of the step."""
        def gnorm(params):
            g = [p.grad.norm() for p in params if p.grad is not None]
            return torch.stack(g).norm().item() if g else 0.0
        pol = gnorm(self.policy.parameters())
        head = gnorm(self.latent_head.parameters()) if self.latent_head is not None else 0.0
        info = (f"[nan-debug] step {self.n_steps} loss {loss.item():.4f} grad norm policy {pol:.1f} head {head:.1f} "
                f"max|offset| {getattr(self, 'dbg_off_max', float('nan')):.2f} max|seed latent| "
                f"{getattr(self, 'dbg_seed_max', float('nan')):.2f}")
        finite = math.isfinite(loss.item()) and math.isfinite(pol) and math.isfinite(head)
        if not finite or self.n_steps % 50 == 0:
            print(info, getattr(self, "dbg_parts", {}), flush=True)
        if not finite:
            raise RuntimeError("[nan-debug] first non-finite loss or gradient")

    def _optimizer_step(self, loss, logged_loss):
        """Logged (train loss in the epoch lines): the policy loss without the auxiliary latent prediction."""
        (self.loss_scale * loss).backward()
        if os.environ.get("SDR_NAN_DEBUG"):
            self._nan_debug(loss)
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.train.grad_clip)
        self.optimizer.step()
        if self.target_encoder is not None:
            self._update_target_encoder()
        return logged_loss.item()

    def observe(self, data):
        partner = data.pop("partner", None)
        if self.seed_loader is None:
            if partner is None:
                return Sequential.observe(self, data)
            self.n_steps += 1
            data = self.map_tensor_to_device(data)
            self.optimizer.zero_grad()
            loss = self.policy.compute_loss(data)
            return self._optimizer_step(self._add_latent_pred_loss(loss, data, partner), loss)
        cur = data
        mode = self.cfg.lifelong.drift_compensation
        per_step = mode.endswith("_grad")
        if not per_step and self.n_steps > 0 and self.n_steps % self.cfg.lifelong.refit_every == 0:
            self._refit()
        self.n_steps += 1
        buf = next(self.seed_loader)
        sidx = buf.pop("seed_index").to(self.cfg.device)
        nb_obs = buf.pop("neighbor_obs", None)
        nb_index = buf.pop("neighbor_index", None)
        nb_index = None if nb_index is None else nb_index.to(self.cfg.device)
        n_cur = data["actions"].shape[0]
        data = self.map_tensor_to_device(merge_datas(data, buf))
        self.optimizer.zero_grad()
        if per_step:
            policy = self.policy
            batch = policy._dp_batch(data)
            nobs = policy.dp.normalizer.normalize(batch["obs"])
            cond = policy.dp.obs_encoder(nobs)  # current + seeds
            cnt = self.seed_count[sidx]
            valid = cnt > 0  # seeds whose cluster has non-seed members
            if mode == "raw_grad":  # plain ER on the raw seeds, no latent replay
                valid = torch.zeros_like(valid)
            r = (torch.rand(len(sidx), generator=self.lat_gen, device=sidx.device) * cnt.clamp(min=1)).long()
            mem = self.member_order[(self.seed_start[sidx] + r)[valid]]
            seed_img = cond[n_cur:, : self._image_dim()][valid]
            # the member's task embedding part is its seed's from this step (same task): with
            # policy.project_language the projection keeps training and the stored one goes stale
            P = self._prop_dim()
            low = torch.cat([self.lat_low[mem][:, :P], cond[n_cur:, self._image_dim() + P:][valid]], 1)
            lat_act = self.lat_actions[mem]
            if mode == "gen_grad":
                off, a_gen, p_gen = self._generate(mem)
                if a_gen is not None:  # joint: generated action chunk and proprio
                    lat_act = a_gen.reshape(lat_act.shape)
                    keep = self.cfg.lifelong.get("gen_action_dct_keep", 0)
                    if keep > 0:
                        lat_act = smooth_action_chunks(lat_act, keep)
                    low = torch.cat([p_gen, low[:, self._prop_dim():]], 1)
            elif mode == "pred_grad":
                R = self._image_dim()
                if self._ar_len() > 0:  # chained steps from the seed to the member
                    z0 = seed_img.detach()
                    x_g = self.old_base[self.lat_task[mem]] + self.lat_sample[mem]
                    z, _ = self._ar_chain(z0, self.seed_g[self.lat_seed[mem]], x_g, self.old_prop, self.old_steps,
                                          to_last_step=False)
                    off = z - z0
                else:
                    c = self._flow_cond(seed_img.detach(), cond[n_cur:, R:R + P][valid].detach(),
                                        self.lat_low[mem][:, :P], self.lat_path[mem], self.lat_dsteps[mem])
                    off = self.latent_head.sample(c, self.cfg.lifelong.get("latent_pred_sample_steps", 10), self.lat_gen)
                if os.environ.get("SDR_NAN_DEBUG"):
                    self.dbg_off_max = off.abs().max().item()
                    self.dbg_seed_max = seed_img.detach().abs().max().item()
            else:
                off = self._step_offsets(mode, mem)
            lat = torch.cat([seed_img + off, low], dim=1)
            global_cond = torch.cat([cond, lat], dim=0)
            actions = torch.cat([batch["action"], lat_act], dim=0)
            loss = policy.dp.compute_loss_from_global_cond(global_cond, policy.dp.normalizer["action"].normalize(actions))
            if os.environ.get("SDR_NAN_DEBUG"):
                self.dbg_parts = {"policy_loss": loss.item(), "cond_finite": bool(torch.isfinite(cond).all()),
                                  "cond_max": cond.detach().abs().max().item(),
                                  "replayed_finite": bool(torch.isfinite(lat).all()),
                                  "actions_norm_max": policy.dp.normalizer["action"].normalize(actions).abs().max().item()}
            w = self.cfg.lifelong.get("seed_distill_weight", 0.0)
            if w > 0:
                distill = self._seed_distill_loss(nobs, n_cur, sidx)
                if self.n_steps == 1:
                    print(f"[info] SeedDriftER: seed distillation loss at the first step of the task {distill.item():.4f}")
                loss = loss + w * distill
            total = loss
            if self._flow() and (nb_obs is not None or self.sr_cond is not None):
                old_loss = self._old_flow_loss(data, cond, n_cur, sidx, nb_obs, nb_index)
                if os.environ.get("SDR_NAN_DEBUG"):
                    self.dbg_parts["old_flow_loss"] = old_loss.item()
                total = loss + self.cfg.lifelong.latent_pred_weight * old_loss
        else:
            idx = self._next_latent_indices()
            cond = torch.cat([self.lat_img[idx], self.lat_low[idx]], dim=1)
            loss = self.policy.compute_loss_with_latent_replay(data, cond, self.lat_actions[idx])
            total = loss
        if partner is None:
            return self._optimizer_step(total, loss)
        return self._optimizer_step(self._add_latent_pred_loss(total, cur, partner), loss)
