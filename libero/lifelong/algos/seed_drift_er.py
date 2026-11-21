import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from libero.lifelong.algos.base import Sequential
from libero.lifelong.algos.er import ER, merge_datas
from libero.lifelong.algos.offset_generator import OffsetGenerator
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


def temporal_seeds(demo, ts, n_seeds):
    """n_seeds seeds spread over the demos in proportion to their length; each the middle frame of an equal
    contiguous segment of its demo. Returns seed indices and the segment (seed) of every sample."""
    n_demo = int(demo.max()) + 1
    lens = torch.bincount(demo, minlength=n_demo).float()
    per = torch.clamp(torch.round(lens / lens.sum() * n_seeds), min=1).long()
    seeds, assign = [], torch.empty_like(demo)
    start = 0
    for k in range(n_demo):
        L, m = int(lens[k]), int(per[k])
        seg = torch.clamp((torch.arange(L) * m) // L, max=m - 1)
        for q in range(m):
            idx = torch.nonzero(seg == q).squeeze(1)
            seeds.append(start + int(idx[len(idx) // 2]))
            assign[start + idx] = len(seeds) - 1
        start += L
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
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.store = []  # per past task: dict(cond, actions, seeds, assign, dt)
        self.generators = []  # per past task: OffsetGenerator (lifelong.generator != none)
        self.seed_loader = None
        self.n_steps = 0
        self.replay_cost = {}

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

    def end_task(self, dataset, task_id, benchmark, env=None):
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
            if self.cfg.lifelong.get("seed_selection", "kmeans") == "temporal":
                seeds, assign = temporal_seeds(demo, ts, k)
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
            self.generators.append(self._fit_generator(entry, task_id))

    def _gen_inputs(self, z_seed, prop, act_flat, dt):
        """Generator condition: seed latent F_j(s), (proprio, action if conditional), dt."""
        parts = [z_seed]
        if self.cfg.lifelong.generator == "cond":
            parts += [prop, act_flat]
        parts.append(dt[:, None])
        return torch.cat(parts, 1)

    def _fit_generator(self, e, task_id):
        R, P = self._image_dim(), self._prop_dim()
        lc = self.cfg.lifelong
        path = os.path.join(self.experiment_dir, "seed_drift", f"gen_task{task_id}.pt")
        nonseed = torch.ones(e["cond"].shape[0], dtype=torch.bool, device=e["cond"].device)
        nonseed[e["seeds"]] = False
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
        self.seed_loader = None
        if self.current_task <= 0:
            return
        assert len(self.store) == len(self.datasets) == task, "[error] SeedDriftER store out of sync"
        dev = self.cfg.device
        bs = self.cfg.train.batch_size
        # raw seeds of all past tasks: replay loader (encoder gradient) and a cached copy for refits
        seed_sets = [Subset(ds, e["seeds"].cpu().tolist()) for ds, e in zip(self.datasets, self.store)]
        seed_all = torch.utils.data.ConcatDataset(seed_sets)
        self.seed_loader = iter(DataLoader(
            Indexed(seed_all), batch_size=max(1, round(self.cfg.lifelong.replay_ratio * bs)),
            num_workers=self.cfg.train.num_workers, sampler=EndlessRandomSampler(len(seed_all), self.cfg.seed + task)))
        self.seed_batches = [self.map_tensor_to_device(d) for d in
                             DataLoader(seed_all, batch_size=128, shuffle=False, num_workers=self.cfg.train.num_workers)]

        # replayed latents: the non-seed samples of all past tasks
        R = self._image_dim()
        off, low, acts, task_of, seed_of = [], [], [], [], []
        seed_base = 0
        self.seed_task, self.seed_stale = [], []
        for j, e in enumerate(self.store):
            nonseed = torch.ones(e["cond"].shape[0], dtype=torch.bool, device=dev)
            nonseed[e["seeds"]] = False
            img = e["cond"][:, :R]
            s_img = img[e["seeds"]]
            off.append((img - s_img[e["assign"]])[nonseed])
            low.append(e["cond"][nonseed, R:])
            acts.append(e["actions"][nonseed])
            task_of.append(torch.full((int(nonseed.sum()),), j, device=dev))
            seed_of.append(e["assign"][nonseed] + seed_base)
            self.seed_task.append(torch.full((len(e["seeds"]),), j, device=dev))
            self.seed_stale.append(s_img)
            seed_base += len(e["seeds"])
        self.lat_off, self.lat_low = torch.cat(off), torch.cat(low)
        self.lat_actions, self.lat_task, self.lat_seed = torch.cat(acts), torch.cat(task_of), torch.cat(seed_of)
        self.lat_dt = torch.cat([e["dt"][self._nonseed(e)] for e in self.store])
        self.seed_task, self.seed_stale = torch.cat(self.seed_task), torch.cat(self.seed_stale)
        self.lat_gen = torch.Generator(device=dev).manual_seed(self.cfg.seed + task)
        # non-seed members of every seed's cluster (seed_offset_grad draws one per replayed seed)
        self.member_order = torch.argsort(self.lat_seed)
        self.seed_count = torch.bincount(self.lat_seed, minlength=len(self.seed_stale))
        self.seed_start = torch.cumsum(self.seed_count, 0) - self.seed_count
        self.lat_perm, self.lat_pos = None, 0
        self.lat_bs = max(1, round(self.cfg.lifelong.latent_replay_ratio * bs))
        self.n_steps = 0
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
        if self.generators:
            gen_b = sum(g.storage_bytes() for g in self.generators)
            side = 0 if self.cfg.lifelong.generator == "joint" else 2 * (self._prop_dim() + self.lat_actions[0].numel())
            self.replay_cost.update({
                "generator_bytes_fp16": int(gen_b), "side_bytes_per_sample_fp16": int(side),
                "buffer_bytes_generator": int(n_seed * raw_b + n_seed * 2 * R + n_lat * (side + 2) + gen_b)})
            print(f"[info] SeedDriftER generator storage: {gen_b / 1e6:.1f} MB generators + {side} B/sample "
                  f"side info, total {self.replay_cost['buffer_bytes_generator'] / 1e6:.1f} MB")
        print(f"[info] SeedDriftER: {n_seed} raw seeds + {n_lat} latents of {task} past tasks, "
              f"compensation {self.cfg.lifelong.drift_compensation}, "
              f"{self.replay_cost['buffer_bytes'] / 1e6:.1f} MB (all raw: "
              f"{self.replay_cost['buffer_bytes_if_all_raw'] / 1e6:.1f} MB)")

    @staticmethod
    def _nonseed(e):
        m = torch.ones(e["cond"].shape[0], dtype=torch.bool, device=e["cond"].device)
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
        if mode == "seed_offset_grad":
            return self.lat_off[mem]
        if mode == "seed_only_grad":
            return torch.zeros_like(self.lat_off[mem])
        if mode == "noise_grad":
            d = self.lat_off[mem]
            e = torch.randn(d.shape, device=d.device, generator=self.lat_gen)
            return e / e.norm(dim=1, keepdim=True) * d.norm(dim=1, keepdim=True)
        raise ValueError(f"[error] unknown lifelong.drift_compensation {mode}")

    @torch.no_grad()
    def _generate(self, mem):
        """Generated offsets, flat actions and proprio (the latter two None for conditional generators)."""
        R, P = self._image_dim(), self._prop_dim()
        joint = self.cfg.lifelong.generator == "joint"
        off = torch.empty_like(self.lat_off[mem])
        act = torch.empty(len(mem), self.lat_actions[0].numel(), device=off.device) if joint else None
        prop = torch.empty(len(mem), P, device=off.device) if joint else None
        task = self.lat_task[mem]
        for j in torch.unique(task).tolist():
            m = task == j
            mm = mem[m]
            C = self._gen_inputs(self.seed_stale[self.lat_seed[mm]], self.lat_low[mm][:, :P],
                                 self.lat_actions[mm].reshape(len(mm), -1), self.lat_dt[mm])
            d, a, p = self.generators[j].sample(C, self.cfg.lifelong.gen_sample_steps, self.lat_gen)
            off[m] = d
            if joint:
                act[m], prop[m] = a, p
        return off, act, prop

    def observe(self, data):
        if self.seed_loader is None:
            return Sequential.observe(self, data)
        mode = self.cfg.lifelong.drift_compensation
        per_step = mode.endswith("_grad")
        if not per_step and self.n_steps > 0 and self.n_steps % self.cfg.lifelong.refit_every == 0:
            self._refit()
        self.n_steps += 1
        buf = next(self.seed_loader)
        sidx = buf.pop("seed_index").to(self.cfg.device)
        n_cur = data["actions"].shape[0]
        data = self.map_tensor_to_device(merge_datas(data, buf))
        self.optimizer.zero_grad()
        if per_step:
            policy = self.policy
            batch = policy._dp_batch(data)
            cond = policy.dp.obs_encoder(policy.dp.normalizer.normalize(batch["obs"]))  # current + seeds
            cnt = self.seed_count[sidx]
            valid = cnt > 0  # seeds whose cluster has non-seed members
            r = (torch.rand(len(sidx), generator=self.lat_gen, device=sidx.device) * cnt.clamp(min=1)).long()
            mem = self.member_order[(self.seed_start[sidx] + r)[valid]]
            seed_img = cond[n_cur:, : self._image_dim()][valid]
            low, lat_act = self.lat_low[mem], self.lat_actions[mem]
            if mode == "gen_grad":
                off, a_gen, p_gen = self._generate(mem)
                if a_gen is not None:  # joint: generated action chunk and proprio
                    lat_act = a_gen.reshape(lat_act.shape)
                    low = torch.cat([p_gen, low[:, self._prop_dim():]], 1)
            else:
                off = self._step_offsets(mode, mem)
            lat = torch.cat([seed_img + off, low], dim=1)
            global_cond = torch.cat([cond, lat], dim=0)
            actions = torch.cat([batch["action"], lat_act], dim=0)
            loss = policy.dp.compute_loss_from_global_cond(global_cond, policy.dp.normalizer["action"].normalize(actions))
        else:
            idx = self._next_latent_indices()
            cond = torch.cat([self.lat_img[idx], self.lat_low[idx]], dim=1)
            loss = self.policy.compute_loss_with_latent_replay(data, cond, self.lat_actions[idx])
        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.train.grad_clip)
        self.optimizer.step()
        return loss.item()
