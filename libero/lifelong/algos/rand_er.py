import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, Subset

from libero.lifelong.algos.base import Sequential
from libero.lifelong.algos.er import ER, cycle


class RandER(ER):
    """
    Experience replay with a buffer of a fixed fraction of all past task data, as
    RandER in the Memory Anchors paper (Sec. 3.1, App. B.3): the buffer holds
    buffer_fraction * (number of past transitions) transitions, sampled uniformly
    without replacement from all past tasks, and is resampled at every new task.
    lifelong.n_memories (if set) is a fixed buffer size instead.

    Unlike ER (the first n_memories sequences of every past task), the buffer size
    grows with the amount of past data and is not split evenly across tasks.
    Every training batch is concatenated with a buffer batch of replay_ratio times its size
    (1.0: the same size, as ER).
    """

    def start_task(self, task):
        # skip ER.start_task, which builds the truncated per-task buffers
        Sequential.start_task(self, task)
        if self.current_task > 0:
            past_data = ConcatDataset(self.datasets)
            n_memories = self.cfg.lifelong.get("n_memories")
            if n_memories is None:
                n_memories = max(1, round(self.cfg.lifelong.buffer_fraction * len(past_data)))
            assert n_memories <= len(past_data), "[error] buffer larger than the past data"
            rng = np.random.default_rng(self.cfg.seed + task)
            indices = self.select_buffer(past_data, n_memories, rng)
            buf = Subset(past_data, indices.tolist())
            print(
                f"[info] {self.cfg.lifelong.algo} buffer: {n_memories} of {len(past_data)} "
                f"past transitions ({n_memories / len(past_data):.2%})"
            )
            replay_batch_size = max(
                1, round(self.cfg.lifelong.get("replay_ratio", 1.0) * self.cfg.train.batch_size)
            )
            self.buffer = cycle(
                DataLoader(
                    buf,
                    batch_size=replay_batch_size,
                    num_workers=self.cfg.train.num_workers,
                    sampler=RandomSampler(buf),
                    persistent_workers=True,
                )
            )

    def select_buffer(self, past_data, n_memories, rng):
        """Indices into past_data (ConcatDataset of the past tasks) of the replay buffer."""
        return rng.choice(len(past_data), size=n_memories, replace=False)
