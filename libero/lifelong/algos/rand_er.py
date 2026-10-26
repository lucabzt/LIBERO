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

    Unlike ER (the first n_memories sequences of every past task), the buffer size
    grows with the amount of past data and is not split evenly across tasks.
    Like ER, every training batch is concatenated with a buffer batch of the same size.
    """

    def start_task(self, task):
        # skip ER.start_task, which builds the truncated per-task buffers
        Sequential.start_task(self, task)
        if self.current_task > 0:
            past_data = ConcatDataset(self.datasets)
            n_memories = max(1, round(self.cfg.lifelong.buffer_fraction * len(past_data)))
            rng = np.random.default_rng(self.cfg.seed + task)
            indices = rng.choice(len(past_data), size=n_memories, replace=False)
            buf = Subset(past_data, indices.tolist())
            print(
                f"[info] RandER buffer: {n_memories} of {len(past_data)} past transitions "
                f"({self.cfg.lifelong.buffer_fraction:.2%})"
            )
            self.buffer = cycle(
                DataLoader(
                    buf,
                    batch_size=self.cfg.train.batch_size,
                    num_workers=self.cfg.train.num_workers,
                    sampler=RandomSampler(buf),
                    persistent_workers=True,
                )
            )
