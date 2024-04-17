from typing import TypeVar, Optional, Iterator
import torch.utils.data.distributed
from torch.utils.data.distributed import Sampler, Dataset
from utils.log import log
import math
T_co = TypeVar('T_co', covariant=True)


class DistributedPartitionSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0) -> None:
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=True)

    def __iter__(self):
        # log.debug(f'shuffle: {self.shuffle}')
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # if not self.drop_last:
        #     # add extra samples to make it evenly divisible
        #     padding_size = self.total_size - len(indices)
        #     if padding_size <= len(indices):
        #         indices += indices[:padding_size]
        #     else:
        #         indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        # else:
        # remove tail of data to make it evenly divisible.
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.num_samples * self.rank:self.num_samples * (self.rank+1)]
        # log.debug(f"{len(indices)} {self.num_samples} {self.rank} {self.num_replicas}")
        # log.debug(f"{indices}")
        assert len(indices) == self.num_samples
        return iter(indices)
