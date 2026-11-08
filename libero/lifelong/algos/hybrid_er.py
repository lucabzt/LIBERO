import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from libero.lifelong.algos.anchor_er import AnchorER
from libero.lifelong.algos.er import merge_datas
from libero.lifelong.algos.rand_er import EndlessRandomSampler


class FlaggedSubset(Dataset):
    """Subset of the past data whose samples carry replay_anchor (1.0 for anchors, else 0.0)."""

    def __init__(self, dataset, indices, anchor_mask):
        self.dataset = dataset
        self.indices = list(indices)
        self.anchor_mask = anchor_mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        data = self.dataset[self.indices[i]]
        data["replay_anchor"] = torch.tensor(float(self.anchor_mask[i]))
        return data


class HybridER(AnchorER):
    """
    HybridER: AnchorER where the obs encoder keeps training, but only on the current task and on
    the anchors. At every task the anchors (anchor_fraction of the buffer, AnchorER selection) are
    replayed raw and send gradients into the encoder; all other buffer samples are replayed as
    latents, here recomputed with the current encoder at every step under no_grad (fresh latents,
    the upper bound of storing latents once and correcting their drift). The U-Net trains on all
    samples. Storage is therefore that of raw replay of the whole buffer; a version with stored
    latents would need anchors raw + the rest as latents (task<k>_cost.json: buffer_bytes_hybrid).

    Typical use: buffer_fraction=1.0 with pool_n_per_task=1000 (same buffer as LIBERO ER) or with
    the full past data (same buffer as RandER 100%).
    """

    def start_task(self, task):
        super().start_task(task)  # AnchorER selection + raw replay loader of the buffer
        if self.current_task <= 0:
            return
        assert self.cfg.lifelong.mode == "add" and self.last_anchors is not None
        assert not self.cfg.lifelong.get("latent_replay", False)
        assert not self.policy.obs_encoder_frozen(), "[error] HybridER trains the obs encoder"
        past_data = self.past_pool()
        indices = self.last_buffer_indices
        anchor_set = set(int(i) for i in self.last_anchors)
        anchor_mask = np.array([int(i) in anchor_set for i in indices])
        buf = FlaggedSubset(past_data, indices, anchor_mask)
        replay_batch_size = max(
            1, round(self.cfg.lifelong.get("replay_ratio", 1.0) * self.cfg.train.batch_size)
        )
        self.buffer = iter(
            DataLoader(
                buf,
                batch_size=replay_batch_size,
                num_workers=self.cfg.train.num_workers,
                sampler=EndlessRandomSampler(len(buf), self.cfg.seed + task),
            )
        )
        n_anchor = int(anchor_mask.sum())
        raw_b = self.replay_cost["raw_bytes_per_sample"]
        lat_b = self.replay_cost["latent_bytes_per_sample"]
        self.replay_cost.update({
            "replay": "hybrid (anchors raw + encoder grad, rest recomputed latents)",
            "n_anchor_raw": n_anchor,
            "n_latent": int(len(indices) - n_anchor),
            "buffer_bytes_hybrid": int(n_anchor * raw_b + (len(indices) - n_anchor) * lat_b),
        })
        print(
            f"[info] HybridER buffer: {n_anchor} anchors raw (encoder gradient), "
            f"{len(indices) - n_anchor} as recomputed latents; stored-latent version would need "
            f"{self.replay_cost['buffer_bytes_hybrid'] / 1e6:.1f} MB vs "
            f"{self.replay_cost['buffer_bytes_if_raw'] / 1e6:.1f} MB raw"
        )

    def select_buffer(self, past_data, n_memories, rng):
        indices = super().select_buffer(past_data, n_memories, rng)
        self.last_buffer_indices = indices
        return indices

    def observe(self, data):
        if self.current_task <= 0 or self.buffer is None:
            return super().observe(data)
        buf_data = next(self.buffer)
        anchor = buf_data.pop("replay_anchor")
        n_cur = data["actions"].shape[0]
        data = merge_datas(data, buf_data)
        grad_mask = torch.cat([torch.ones(n_cur), anchor]).bool()
        data = self.map_tensor_to_device(data)
        grad_mask = grad_mask.to(self.cfg.device)

        self.optimizer.zero_grad()
        loss = self.policy.compute_loss_encoder_grad_mask(data, grad_mask)
        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.train.grad_clip)
        self.optimizer.step()
        return loss.item()
