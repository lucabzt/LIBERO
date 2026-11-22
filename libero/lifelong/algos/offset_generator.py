"""
Per-task generator of seed offsets for SeedDriftER (lifelong.generator), trained at the end of a task on
its own stored latents and frozen afterwards. Conditional rectified flow (flow matching) in the whitened
PCA space of the task's offsets d(x) = F_j(x) - F_j(s(x)).

  conditional (joint=False): samples d ~ G(. | F_j(s), proprio(x), action(x), dt(x)); action and proprio
                             are stored per sample and used as replay labels / condition
  joint (joint=True):        samples [d, action, proprio] ~ G(. | F_j(s), dt(x)); nothing stored per sample
                             (a temporal-seed sample is named by its seed and dt)
Offline study: experiments/2026-11-21_offset_generator_study (gen_study.py, same model).
SharedOffsetGenerator: one conditional generator for all past tasks, trained continually (shared_study.py).
"""
import math

import torch
import torch.nn as nn


def time_emb(t, dim=32):
    f = torch.exp(torch.linspace(0, math.log(1000), dim // 2, device=t.device))
    return torch.cat([torch.sin(t[:, None] * f), torch.cos(t[:, None] * f)], 1)


def smooth_action_chunks(a, keep):
    """
    Action chunks (B, T, 7) with the arm dims low-passed along the chunk (first `keep` DCT coefficients) and the
    gripper snapped to +-1. Generated chunks carry independent noise per timestep and a blurred gripper command
    (experiments/2026-11-21_generator_failure); recorded chunks are smooth and their gripper is +-1.
    """
    T = a.shape[1]
    t = torch.arange(T, device=a.device, dtype=a.dtype)
    D = torch.cos(math.pi / T * (t[None, :] + 0.5) * t[:, None])
    D = (D / D.norm(dim=1, keepdim=True))[:keep]
    arm = torch.einsum("kt,bkd->btd", D, torch.einsum("kt,btd->bkd", D, a[..., :6]))
    return torch.cat([arm, torch.sign(a[..., 6:])], -1)


class ResMLP(nn.Module):
    def __init__(self, d_in, d_cond, d_out, width, blocks=4):
        super().__init__()
        self.inp = nn.Linear(d_in + d_cond, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                                   nn.Linear(width, width)) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d_out))

    def forward(self, x, c):
        h = self.inp(torch.cat([x, c], 1))
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


def train_flow(model, Y, C, steps, lr=3e-4, bs=512, seed=0):
    """Rectified-flow training of model.net on targets Y given conditions C, from the model's current weights."""
    g = torch.Generator(device=Y.device).manual_seed(seed)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    model.train()
    for _ in range(steps):
        i = torch.randint(0, len(Y), (bs,), device=Y.device, generator=g)
        y = Y[i]
        z = torch.randn(y.shape, device=Y.device, generator=g)
        t = torch.rand(len(y), device=Y.device, generator=g)
        xt = (1 - t[:, None]) * z + t[:, None] * y
        v = model.net(torch.cat([xt, time_emb(t)], 1), model._cond(C[i]))
        loss = ((v - (y - z)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    return loss.item()


@torch.no_grad()
def flow_sample(model, C, d_out, n_steps, generator=None):
    x = torch.randn((len(C), d_out), device=C.device, generator=generator)
    ce = model._cond(C)
    for k in range(n_steps):
        t = torch.full((len(C),), k / n_steps, device=C.device)
        x = x + model.net(torch.cat([x, time_emb(t)], 1), ce) / n_steps
    return x


class OffsetGenerator(nn.Module):
    def __init__(self, R, d_cond_rest, n_pca, n_act, n_prop, joint, width, d_seed=256):
        super().__init__()
        self.R, self.n_pca, self.n_act, self.n_prop, self.joint = R, n_pca, n_act, n_prop, joint
        self.d_out = n_pca + (n_act + n_prop if joint else 0)
        self.seed_proj = nn.Linear(R, d_seed)
        self.net = ResMLP(self.d_out + 32, d_seed + d_cond_rest, self.d_out, width)
        # PCA basis and normalizers (buffers: saved with the generator, count toward its storage)
        self.register_buffer("mu", torch.zeros(R))
        self.register_buffer("V", torch.zeros(n_pca, R))
        self.register_buffer("scale", torch.ones(n_pca))
        self.register_buffer("a_lo", torch.zeros(n_act))
        self.register_buffer("a_hi", torch.ones(n_act))
        self.register_buffer("p_mu", torch.zeros(n_prop))
        self.register_buffer("p_sd", torch.ones(n_prop))

    def _cond(self, c):
        return torch.cat([self.seed_proj(c[:, : self.R]), c[:, self.R:]], 1)

    def _target(self, d, a, p):
        y = ((d - self.mu) @ self.V.T) / self.scale
        if self.joint:
            a = 2 * (a - self.a_lo) / (self.a_hi - self.a_lo).clamp(min=1e-6) - 1
            y = torch.cat([y, a, (p - self.p_mu) / self.p_sd], 1)
        return y

    def fit(self, d, a, p, C, steps, lr=3e-4, bs=512, seed=0):
        """d (N, R) offsets, a (N, n_act) flat actions, p (N, n_prop) proprio, C (N, R + d_cond_rest)."""
        with torch.no_grad():
            self.mu.copy_(d.mean(0))
            _, S, Vh = torch.linalg.svd(d - self.mu, full_matrices=False)
            k = min(self.n_pca, Vh.shape[0])
            self.V.zero_()
            self.V[:k] = Vh[:k]
            self.scale.copy_(torch.cat([S[:k] / math.sqrt(len(d)), torch.ones(self.n_pca - k, device=d.device)]).clamp(min=1e-6))
            self.a_lo.copy_(a.min(0).values)
            self.a_hi.copy_(a.max(0).values)
            self.p_mu.copy_(p.mean(0))
            self.p_sd.copy_(p.std(0).clamp(min=1e-6))
            Y = self._target(d, a, p)
        return train_flow(self, Y, C, steps, lr, bs, seed)

    @torch.no_grad()
    def sample(self, C, n_steps=20, generator=None):
        """Offsets (B, R) and, if joint, flat actions (B, n_act) and proprio (B, n_prop); else None, None."""
        x = flow_sample(self, C, self.d_out, n_steps, generator)
        d = (x[:, : self.n_pca] * self.scale) @ self.V + self.mu
        if not self.joint:
            return d, None, None
        a = (x[:, self.n_pca: self.n_pca + self.n_act] + 1) / 2 * (self.a_hi - self.a_lo) + self.a_lo
        p = x[:, self.n_pca + self.n_act:] * self.p_sd + self.p_mu
        return d, a, p

    def storage_bytes(self):
        """fp16 storage of all parameters and buffers."""
        return 2 * (sum(q.numel() for q in self.parameters()) + sum(b.numel() for b in self.buffers()))


class SharedOffsetGenerator(nn.Module):
    """
    One conditional generator of seed offsets for all past tasks (lifelong.gen_shared), trained continually: at
    the end of task j on task j's offsets plus its own samples for all earlier tasks (self-replay; their conditions
    are stored), task j's real samples repeated to about new_task_share of the mix. Offsets are generated in
    per-task whitened PCA bases (fit when the task ends, 2 * n_pca * R bytes each), or with n_pca=0 directly in the
    offset space, standardized per dimension with the first task's statistics (no bases; works for small latents such
    as the 256-d spatial softmax one, not for 2048-d); the condition carries the task embedding.
    Offline: experiments/2026-11-21_offset_generator_study/shared_study.py --continual (--pca_per_task | --pca 0).
    """

    def __init__(self, R, d_cond_rest, n_pca, width, d_seed=256):
        super().__init__()
        self.R, self.n_pca, self.n_tasks = R, n_pca, 0
        self.d_out = n_pca if n_pca > 0 else R
        self.seed_proj = nn.Linear(R, d_seed)
        self.net = ResMLP(self.d_out + 32, d_seed + d_cond_rest, self.d_out, width)

    def _cond(self, c):
        return torch.cat([self.seed_proj(c[:, : self.R]), c[:, self.R:]], 1)

    def _register_basis(self, mu, V, scale):
        j = self.n_tasks
        self.register_buffer(f"mu{j}", mu)
        self.register_buffer(f"V{j}", V)
        self.register_buffer(f"scale{j}", scale)
        self.n_tasks += 1

    def add_basis(self, d):
        """Whitened PCA basis of a new task's offsets d (N, R); with n_pca=0 the first task's standardization."""
        if self.n_pca == 0:
            if self.n_tasks == 0:
                self.register_buffer("std_mu", d.mean(0))
                self.register_buffer("std_scale", d.std(0).clamp(min=1e-6))
            self.n_tasks += 1
            return
        mu = d.mean(0)
        _, S, Vh = torch.linalg.svd(d - mu, full_matrices=False)
        k = min(self.n_pca, Vh.shape[0])
        V = torch.zeros(self.n_pca, self.R, device=d.device)
        V[:k] = Vh[:k]
        scale = torch.cat([S[:k] / math.sqrt(len(d)), torch.ones(self.n_pca - k, device=d.device)]).clamp(min=1e-6)
        self._register_basis(mu, V, scale)

    def ensure_bases(self, n, device):
        """Placeholder bases so that a saved state with n tasks can be loaded."""
        if self.n_pca == 0:
            if self.n_tasks == 0:
                self.register_buffer("std_mu", torch.zeros(self.R, device=device))
                self.register_buffer("std_scale", torch.ones(self.R, device=device))
            self.n_tasks = max(self.n_tasks, n)
            return
        while self.n_tasks < n:
            self._register_basis(torch.zeros(self.R, device=device), torch.zeros(self.n_pca, self.R, device=device),
                                 torch.ones(self.n_pca, device=device))

    def coords(self, d, j):
        if self.n_pca == 0:
            return (d - self.std_mu) / self.std_scale
        return ((d - getattr(self, f"mu{j}")) @ getattr(self, f"V{j}").T) / getattr(self, f"scale{j}")

    def decode(self, y, j):
        if self.n_pca == 0:
            return y * self.std_scale + self.std_mu
        return (y * getattr(self, f"scale{j}")) @ getattr(self, f"V{j}") + getattr(self, f"mu{j}")

    @torch.no_grad()
    def sample_coords(self, C, n_steps=20, generator=None):
        return flow_sample(self, C, self.d_out, n_steps, generator)

    @torch.no_grad()
    def sample(self, C, j, n_steps=20, generator=None):
        """Offsets (B, R) of task j for conditions C."""
        return self.decode(self.sample_coords(C, n_steps, generator), j)

    def fit_task(self, d, C, old_C, steps, new_task_share, n_steps=20, seed=0):
        """
        Stage of task j = self.n_tasks: d (N, R) its offsets, C (N, ...) their conditions; old_C conditions of all
        earlier tasks' samples (self-replay targets are sampled from the model before this stage).
        """
        Y_old = None
        if old_C is not None:
            g = torch.Generator(device=d.device).manual_seed(seed + 1)
            Y_old = torch.cat([self.sample_coords(old_C[i: i + 4096], n_steps, g) for i in range(0, len(old_C), 4096)])
        self.add_basis(d)
        Y = self.coords(d, self.n_tasks - 1)
        if Y_old is not None:
            reps = max(1, round(new_task_share / (1 - new_task_share) * len(Y_old) / len(Y))) if new_task_share > 0 else 1
            Y, C = torch.cat([Y.repeat(reps, 1), Y_old]), torch.cat([C.repeat(reps, 1), old_C])
        return train_flow(self, Y, C, steps, seed=seed)

    def storage_bytes(self):
        """fp16 storage of all parameters and buffers (network once + per-task bases)."""
        return 2 * (sum(q.numel() for q in self.parameters()) + sum(b.numel() for b in self.buffers()))
