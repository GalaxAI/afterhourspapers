from typing import Generic, Iterator, Optional, TypeVar

from torch import backends, cuda
from torch.utils.data import DataLoader

T = TypeVar("T")  # Generic type for the batch data

def_device = "mps" if backends.mps.is_available() else "cuda" if cuda.is_available() else "cpu"


class StepDataLoader(Generic[T]):
    """
    An iterable that yields batches from a DataLoader across multiple epochs continuously.
    Has a __len__ method, making it compatible with tqdm/trange.

    Args:
        dataloader (DataLoader): PyTorch DataLoader instance.
        num_epochs (int): Number of epochs to iterate over.
    """

    def __init__(self, dataloader: DataLoader, num_epochs: int):
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self._iterator: Optional[Iterator[T]] = None

    def __iter__(self) -> Iterator[T]:
        for _ in range(self.num_epochs):
            yield from self.dataloader

    def __len__(self) -> int:
        return len(self.dataloader) * self.num_epochs

    def next_batch(self) -> T:
        """Get the next batch from the iterator"""
        if self._iterator is None:
            self._iterator = iter(self)
        return next(self._iterator)
