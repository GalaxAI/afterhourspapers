__all__ = ["DataLoaders"]

from operator import itemgetter

from torch.utils.data import DataLoader, default_collate


def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs), DataLoader(valid_ds, batch_size=bs * 2, **kwargs))


def collate_dict(ds):
    get = itemgetter(*ds.features)

    def _f(b):
        return get(default_collate(b))

    return _f


class DataLoaders:
    def __init__(self, *dls):
        self.train, self.valid = dls[:2]

    @classmethod
    def from_dd(cls, dd, batch_size, as_tuple=True, **kwargs):
        f = collate_dict(dd["train"])
        return cls(*get_dls(*dd.values(), bs=batch_size, collate_fn=f, **kwargs))
