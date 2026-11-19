import json
import os
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from libero.lifelong.algos.anchor_er import selection, visualization
from libero.lifelong.algos.base import Sequential
from libero.lifelong.algos.rand_er import RandER
from libero.lifelong.report.anchor_er_report import selection_percentages


class StaleLatentPool:
    """Past data stored only as stale latents (lifelong.stale_latent_replay): per sample the U-Net
    global condition (N, D) computed once right after its task was learned, the unnormalized action
    chunk (N, horizon, action_dim), its task id and timestep in the demo (CPU tensors / arrays)."""

    def __init__(self, cond, actions, task, t):
        self.cond, self.actions, self.task, self.t = cond, actions, task, t

    def __len__(self):
        return len(self.cond)

    @classmethod
    def concat(cls, parts):
        return cls(
            torch.cat([p.cond for p in parts]), torch.cat([p.actions for p in parts]),
            np.concatenate([p.task for p in parts]), np.concatenate([p.t for p in parts]),
        )


class AnchorER(RandER):
    """
    AnchorER from the Memory Anchors paper (Sec. 4, 5.2, App. B.1, Alg. 1): RandER whose
    buffer is partly filled with Memory Anchors, past data that overlaps the new task in the
    policy's observation latent space where the new task requires different actions.

    At the start of every task after the first, with the policy trained up to the last task:
      1. observation overlap: new samples are split into two k-means clusters by their
         distances to all past samples in the obs-encoder latent (all modalities, every
         dimension z-scored with past-data stats); the closer cluster overlaps the past data
      2. action disagreement: overlapping new samples whose noise-denoise disagreement
         exceeds mu + std_cutoff * sigma of the disagreement on held-out past demos
         (data.n_val_demos)
      3. anchors: past samples ranked by the median distance to their n_neighbors nearest
         disagreeing new samples
    mode "add": the anchor_fraction * n_memories best anchors plus uniformly sampled other
    past data fill the buffer (Table 4). mode "remove": the remove_fraction of all past data
    with the best scores is excluded before sampling the buffer uniformly (Sec. 5.1).
    Training is RandER's (concatenated replay batches).

    Selection results and plots are written to <experiment_dir>/anchorER/task<k>/; main.py
    summarizes them in anchorER/report.md (libero/lifelong/report/anchor_er_report.py).

    lifelong.stale_latent_replay (simple latent replay without pretraining or freezing): at the
    start of task k the training and held-out data of task k-1 are encoded once with the policy
    after task k-1 (eval mode, center crop) into U-Net global conditions + action chunks
    (<experiment_dir>/stale_latents/task<k-1>.pt) and never re-encoded. Past data exists only as
    these stale latents: the selection uses them for the past side (representations = the stored
    condition split by obs key, disagreement baseline = held-out latents denoised by the current
    U-Net), and the buffer replays them (RandER's latent replay path: the current-task batch goes
    through the obs encoder, which keeps training, the replayed latents only into the U-Net).
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.current_dataset = None
        self.benchmark = None
        # held-out demos of every task, set by main.py
        self.val_datasets = None

    def learn_one_task(self, dataset, task_id, benchmark, result_summary):
        # start_task only gets the task id, the selection needs the new data
        self.current_dataset = dataset
        self.benchmark = benchmark
        return super().learn_one_task(dataset, task_id, benchmark, result_summary)

    # ------------------------------------------------------------------
    # stale latent replay
    # ------------------------------------------------------------------
    def start_task(self, task):
        cfg = self.cfg.lifelong
        if not cfg.get("stale_latent_replay", False):
            return super().start_task(task)
        assert not cfg.get("latent_replay", False), "[error] stale_latent_replay replaces latent_replay"
        assert cfg.get("n_memories") is None and not cfg.get("buffer_indices_file")
        assert cfg.get("pool_n_per_task") is None
        Sequential.start_task(self, task)  # as RandER: no ER buffers
        if self.current_task == 0:
            return
        self._encode_stale(self.current_task - 1)
        pool = StaleLatentPool.concat([self._load_stale(k)["train"] for k in range(self.current_task)])
        n_memories = max(1, round(cfg.buffer_fraction * len(pool)))
        rng = np.random.default_rng(self.cfg.seed + task)
        indices = torch.as_tensor(self.select_buffer(pool, n_memories, rng), dtype=torch.long)
        device = torch.device(self.cfg.device)
        self.latent_cond = pool.cond[indices].to(device).contiguous()
        self.latent_actions = pool.actions[indices].to(device).contiguous()
        self.latent_generator = torch.Generator(device=self.latent_cond.device)
        self.latent_generator.manual_seed(self.cfg.seed + task)
        self.latent_perm, self.latent_pos = None, 0
        self.replay_batch_size = max(1, round(cfg.get("replay_ratio", 1.0) * self.cfg.train.batch_size))
        self.buffer = None  # observe takes RandER's latent path
        self.replay_cost = {"n_buffer": int(n_memories), "n_past": int(len(pool))}
        self.replay_cost.update(self._buffer_bytes(raw=False))
        self.replay_cost["latent_dim"] = int(self.latent_cond.shape[1])
        print(
            f"[info] AnchorER stale latent buffer: {n_memories} of {len(pool)} past samples, "
            f"global cond dim {self.latent_cond.shape[1]}"
        )

    def _stale_file(self, k):
        return os.path.join(self.experiment_dir, "stale_latents", f"task{k}.pt")

    @torch.no_grad()
    def _encode_part(self, dataset, k):
        conds, actions = [], []
        for data in self._loader(dataset):
            cond, act = self.policy.encode_latents(self.map_tensor_to_device(data))
            conds.append(cond.float().cpu())
            actions.append(act.float().cpu())
        seq = dataset.sequence_dataset
        t = np.array([i - seq._demo_id_to_start_indices[seq._index_to_demo_id[i]] for i in range(len(dataset))])
        return StaleLatentPool(torch.cat(conds), torch.cat(actions), np.full(len(dataset), k), t)

    def _encode_stale(self, k):
        """Stale latents of task k with the current policy (= the policy right after task k, also
        on resume: main.py loads task<k>_model.pth). Skipped if they exist (resume)."""
        path = self._stale_file(k)
        if os.path.exists(path):
            return
        assert all(os.path.exists(self._stale_file(j)) for j in range(k)), (
            f"[error] stale latents of tasks before {k} missing: they can only be encoded right after their task"
        )
        assert self.val_datasets is not None and self.val_datasets[k] is not None, (
            "[error] stale_latent_replay needs held-out demos (data.n_val_demos > 0)"
        )
        t0 = time.time()
        was_training = self.policy.training
        self.policy.eval()  # center crop, as at test time
        try:
            parts = {"train": self._encode_part(self.datasets[k], k),
                     "val": self._encode_part(self.val_datasets[k], k)}
        finally:
            self.policy.train(was_training)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({name: vars(p) for name, p in parts.items()}, path + ".tmp")
        os.replace(path + ".tmp", path)
        print(
            f"[info] stale latents of task {k}: {len(parts['train'])} train + {len(parts['val'])} "
            f"held-out samples, dim {parts['train'].cond.shape[1]}, {time.time() - t0:.0f} s"
        )

    def _load_stale(self, k):
        try:
            d = torch.load(self._stale_file(k), weights_only=False)
        except TypeError:  # torch < 1.13 has no weights_only
            d = torch.load(self._stale_file(k))
        return {name: StaleLatentPool(**v) for name, v in d.items()}

    def _val_data(self, task):
        if self.cfg.lifelong.get("stale_latent_replay", False):
            return StaleLatentPool.concat([self._load_stale(k)["val"] for k in range(task)])
        return ConcatDataset(self.val_datasets[:task])

    def select_buffer(self, past_data, n_memories, rng):
        out_dir = os.path.join(self.experiment_dir, "anchorER", f"task{self.current_task}")
        os.makedirs(out_dir, exist_ok=True)
        self.policy.eval()
        try:
            with torch.no_grad():
                return self._select(past_data, n_memories, rng, out_dir)
        finally:
            self.policy.train()
            torch.cuda.empty_cache()

    def _loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.cfg.lifelong.selection_batch_size,
            shuffle=False,  # the order gives the indices
            num_workers=self.cfg.train.num_workers,
            persistent_workers=False,
        )

    def _representations(self, dataset):
        device = torch.device(self.cfg.lifelong.selection_device)
        if isinstance(dataset, StaleLatentPool):
            # the stored condition is the per-key features concatenated in this order
            dims = self.policy.latent_dims()
            assert sum(dims.values()) == dataset.cond.shape[1]
            parts = torch.split(dataset.cond, list(dims.values()), dim=1)
            return {key: part.float().to(device) for key, part in zip(dims, parts)}
        reprs = {}
        for data in self._loader(dataset):
            data = self.map_tensor_to_device(data)
            for key, value in self.policy.compute_obs_representations(data).items():
                reprs.setdefault(key, []).append(value.float().to(device))
        return {key: torch.cat(values, dim=0) for key, values in reprs.items()}

    def _disagreement(self, dataset):
        cfg = self.cfg.lifelong
        scores = []
        if isinstance(dataset, StaleLatentPool):
            device = torch.device(self.cfg.device)
            bs = cfg.selection_batch_size
            for i in range(0, len(dataset), bs):
                cond = dataset.cond[i:i + bs].to(device)
                actions = dataset.actions[i:i + bs].to(device)
                draws = [
                    self.policy.compute_action_disagreement_from_cond(
                        cond, actions,
                        noise_level=cfg.noise_level,
                        ood_metric=cfg.disagreement_metric,
                        num_inference_steps=cfg.disagreement_inference_steps,
                        action_scale=cfg.action_scale if cfg.disagreement_metric == "se3" else None,
                    )
                    for _ in range(cfg.n_noise_samples)
                ]
                scores.append(torch.stack(draws, dim=0).mean(dim=0).cpu())
            return torch.cat(scores, dim=0).numpy()
        for data in self._loader(dataset):
            data = self.map_tensor_to_device(data)
            draws = [
                self.policy.compute_action_disagreement(
                    data,
                    noise_level=cfg.noise_level,
                    ood_metric=cfg.disagreement_metric,
                    num_inference_steps=cfg.disagreement_inference_steps,
                    action_scale=cfg.action_scale if cfg.disagreement_metric == "se3" else None,
                )
                for _ in range(cfg.n_noise_samples)
            ]
            scores.append(torch.stack(draws, dim=0).mean(dim=0).cpu())
        return torch.cat(scores, dim=0).numpy()

    def _select(self, past_data, n_memories, rng, out_dir):
        cfg = self.cfg.lifelong
        task = self.current_task
        assert hasattr(self.policy, "compute_action_disagreement"), (
            "[error] AnchorER needs a policy with compute_obs_representations and "
            "compute_action_disagreement (DiffusionPolicy)"
        )
        generator = torch.Generator().manual_seed(self.cfg.seed + task)
        info = {"task": task, "n_old": len(past_data), "n_new": len(self.current_dataset)}
        timings = {}

        # observation latents of all past and new samples
        t0 = time.time()
        old_reprs = self._representations(past_data)
        new_reprs = self._representations(self.current_dataset)
        keys = cfg.representation_keys or list(old_reprs.keys())
        if cfg.get("representation_normalization", "zscore") == "zscore":
            old, stats = selection.normalize_representations(old_reprs, keys)
            new, _ = selection.normalize_representations(new_reprs, keys, stats)
        else:
            # "none": the features concatenated as they are (authors' use_select_keys path)
            old = torch.cat([old_reprs[key] for key in keys], dim=1)
            new = torch.cat([new_reprs[key] for key in keys], dim=1)
        info["representation_dims"] = {key: old_reprs[key].shape[1] for key in keys}
        del old_reprs, new_reprs
        timings["representations"] = time.time() - t0

        # step 1: new data overlapping the past data
        t0 = time.time()
        nn_dist = None
        if cfg.use_overlap:
            overlap_mask, nn_dist = selection.overlap_partition(
                old, new, n_init=cfg.kmeans_n_init, generator=generator
            )
            overlap_mask = overlap_mask.cpu().numpy()
            nn_dist = nn_dist.cpu().numpy()
        else:
            overlap_mask = np.ones(len(new), dtype=bool)
        overlap_idx = np.nonzero(overlap_mask)[0]
        info["n_overlap"] = int(len(overlap_idx))
        timings["overlap"] = time.time() - t0

        # step 2: overlapping new data the old policy disagrees with
        t0 = time.time()
        val_scores = overlap_scores = None
        threshold = None
        info["disagreement_fallback"] = False
        if cfg.use_disagreement:
            assert self.val_datasets is not None and all(
                ds is not None for ds in self.val_datasets[:task]
            ), "[error] AnchorER action disagreement needs held-out demos (data.n_val_demos > 0)"
            val_scores = self._disagreement(self._val_data(task))
            overlap_scores = self._disagreement(Subset(self.current_dataset, overlap_idx.tolist()))
            mu, sigma = float(val_scores.mean()), float(val_scores.std(ddof=1))
            threshold = mu + cfg.std_cutoff * sigma
            od_idx = overlap_idx[overlap_scores > threshold]
            info.update(
                {"n_val": int(len(val_scores)), "mu": mu, "sigma": sigma, "threshold": threshold}
            )
            if len(od_idx) == 0:
                print("[warning] AnchorER: no disagreeing new data, using all overlapping data")
                od_idx = overlap_idx
                info["disagreement_fallback"] = True
        else:
            od_idx = overlap_idx
        info["n_disagreement"] = int(len(od_idx))
        timings["disagreement"] = time.time() - t0

        # step 3: past data closest to the disagreeing new data
        t0 = time.time()
        target = new[torch.as_tensor(od_idx, device=new.device)]
        scores = selection.anchor_scores(old, target, cfg.n_neighbors)
        scores = scores.cpu().numpy()
        del old, new, target
        order = np.argsort(scores, kind="stable")
        if cfg.mode == "add":
            n_anchors = round(cfg.anchor_fraction * n_memories)
            anchors = order[:n_anchors]
            others = rng.choice(order[n_anchors:], size=n_memories - n_anchors, replace=False)
            indices = np.concatenate([anchors, others])
        elif cfg.mode == "remove":
            n_removed = round(cfg.remove_fraction * len(past_data))
            anchors = order[:n_removed]  # excluded from the buffer
            pool = order[n_removed:]
            assert len(pool) >= n_memories, "[error] AnchorER removes too much past data"
            indices = rng.choice(pool, size=n_memories, replace=False)
        else:
            raise ValueError(f"[error] unknown AnchorER mode {cfg.mode}")
        timings["anchors"] = time.time() - t0
        # pool indices of the anchors (HybridER replays them differently)
        self.last_anchors = anchors if cfg.mode == "add" else None
        info["n_anchors"] = int(len(anchors))
        info["n_buffer"] = int(len(indices))
        info["mode"] = cfg.mode
        info["timings_s"] = timings

        self._log_selection(
            out_dir, past_data, info, indices, anchors, scores, overlap_mask, od_idx,
            nn_dist, val_scores, overlap_idx, overlap_scores, threshold,
        )
        print(
            f"[info] AnchorER task {task + 1}: {info['n_overlap']}/{info['n_new']} new samples "
            f"overlap ({info['overlap_pct']:.1f}%), {info['n_disagreement']} disagree "
            f"({info['disagreement_pct_of_overlap']:.1f}% of overlap, "
            f"{info['disagreement_pct_of_new']:.1f}% of new), {info['n_anchors']} anchors "
            f"({cfg.mode}, {info['anchor_pct_of_buffer' if cfg.mode == 'add' else 'anchor_pct_of_old']:.1f}% "
            f"of the {'buffer' if cfg.mode == 'add' else 'past data'}), "
            f"{sum(timings.values()) / 60:.1f} min"
        )
        return indices

    def _sample_info(self, datasets):
        """(task, timestep in demo) of every index of datasets (one or a ConcatDataset)."""
        if isinstance(datasets, StaleLatentPool):
            return datasets.task, datasets.t
        tasks, steps = [], []
        if isinstance(datasets, ConcatDataset):
            parts = list(enumerate(datasets.datasets))
        else:
            parts = [(self.current_task, datasets)]
        for k, ds in parts:
            seq = ds.sequence_dataset
            for i in range(len(ds)):
                demo = seq._index_to_demo_id[i]
                tasks.append(k)
                steps.append(i - seq._demo_id_to_start_indices[demo])
        return np.array(tasks), np.array(steps)

    def _log_selection(
        self, out_dir, past_data, info, indices, anchors, scores, overlap_mask, od_idx,
        nn_dist, val_scores, overlap_idx, overlap_scores, threshold,
    ):
        old_task, old_t = self._sample_info(past_data)
        _, new_t = self._sample_info(self.current_dataset)
        task_names = [self.benchmark.get_task(k).language for k in range(self.current_task)]
        info["anchors_per_task"] = np.bincount(old_task[anchors], minlength=len(task_names)).tolist()
        info["buffer_per_task"] = np.bincount(old_task[indices], minlength=len(task_names)).tolist()
        info["old_per_task"] = np.bincount(old_task, minlength=len(task_names)).tolist()
        info.update(selection_percentages(info))
        info["past_tasks"] = task_names
        info["new_task"] = self.benchmark.get_task(self.current_task).language
        with open(os.path.join(out_dir, "selection.json"), "w") as f:
            json.dump(info, f, indent=2)

        arrays = {
            "buffer_indices": indices,
            "anchor_indices": anchors,
            "anchor_scores": scores,
            "old_task": old_task,
            "old_t": old_t,
            "new_t": new_t,
            "overlap_mask": overlap_mask,
            "disagreement_indices": od_idx,
        }
        if nn_dist is not None:
            arrays["new_nn_dist"] = nn_dist
        if val_scores is not None:
            arrays.update({"val_scores": val_scores, "overlap_scores": overlap_scores})
        np.savez_compressed(os.path.join(out_dir, "selection.npz"), **arrays)

        # plots must not end a long training run
        try:
            visualization.plot_partition(out_dir, new_t, overlap_mask, od_idx)
            visualization.plot_anchors(out_dir, old_task[anchors], old_t[anchors], task_names)
            if val_scores is not None:
                visualization.plot_disagreement(out_dir, val_scores, overlap_scores, threshold)
        except Exception as e:
            print(f"[warning] AnchorER: could not plot the selection: {e!r}")
