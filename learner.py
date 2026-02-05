from functools import partial
from operator import attrgetter
from typing import Callable, Mapping

import torch
import torch.nn.functional as F
from torch import nn, optim

from loader import DataLoaders

def_device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


class CancelFitException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class CancelEpochException(Exception):
    pass


class Callback:
    order = 0


def run_cbs(cbs, method_nm, learn=None):
    for cb in sorted(cbs, key=attrgetter("order")):
        method = getattr(cb, method_nm, None)
        if method is not None:
            method(learn)


def to_device(x, device=def_device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, Mapping):
        return {k: v.to(device) for k, v in x.items()}
    return type(x)(to_device(o, device) for o in x)


class DeviceCB(Callback):
    def __init__(self, device=def_device):
        self.device = device

    def before_fit(self, learn):
        if hasattr(learn.model, "to"):
            learn.model.to(self.device)

    def before_batch(self, learn):
        learn.batch = to_device(learn.batch, device=self.device)


class with_cbs:
    def __init__(self, nm):
        self.nm = nm

    def __call__(self, f):
        def _f(o, *args, **kwargs):
            try:
                o.callback(f"before_{self.nm}")
                f(o, *args, **kwargs)
                o.callback(f"after_{self.nm}")
            except globals()[f"Cancel{self.nm.title()}Exception"]:
                pass
            finally:
                o.callback(f"cleanup_{self.nm}")

        return _f


MetricFunc = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class MetricsCB(Callback):
    def __init__(self, *ms: MetricFunc, **metrics: MetricFunc) -> None:
        from copy import copy

        for o in ms:
            metrics[type(o).__name__] = o
        self.metrics: dict[str, MetricFunc] = metrics
        self.all_metrics: dict[str, MetricFunc] = copy(metrics)
        self.metric_values: dict[str, list[float]] = {}

    def before_epoch(self, learn: "Learner") -> None:
        self.metric_values = {name: [] for name in self.all_metrics}

    def after_epoch(self, learn: "Learner") -> None:
        parts = []
        for m_name, values in self.metric_values.items():
            if not values:
                continue
            mean_value = sum(values) / len(values)
            parts.append(f"{m_name}: {mean_value:.4f}")
        if parts:
            print(f"{self.__class__.__name__} - {', '.join(parts)}")

    def after_batch(self, learn: "Learner") -> None:
        batch = learn.batch[-1]
        for m_name, m_func in self.all_metrics.items():
            value = m_func(learn.preds, batch)
            self.metric_values[m_name].append(float(value))


class ProgressCB(Callback):
    order = Callback.order + 1

    def __init__(self, plot=False):
        self.plot = plot

    def before_epoch(self, learn):
        from helpers import tqdm

        total = len(learn.dl) if hasattr(learn.dl, "__len__") else None
        learn.dl = tqdm(learn.dl, total=total)

    def after_batch(self, learn):
        if hasattr(learn.dl, "set_description"):
            learn.dl.set_description(f"Epoch:{learn.epoch} - {'Train' if learn.training else 'Valid'} Loss: {learn.loss:.3f}")


class Learner:
    def __init__(self, model, dls: DataLoaders, loss_func=F.mse_loss, lr=0.1, cbs=None, opt_func=optim.SGD):
        self.cbs = list(cbs) if cbs else []
        self.model = model
        self.dls = dls
        self.loss_func = loss_func
        self.lr = lr
        self.opt_func = opt_func

    @with_cbs("batch")
    def _one_batch(self):
        self.predict()
        self.callback("after_predict")
        self.get_loss()
        self.callback("after_loss")
        if self.training:
            self.backward()
            self.callback("after_backward")
            self.step()
            self.callback("after_step")
            self.zero_grad()

    @with_cbs("epoch")
    def _one_epoch(self):
        for self.iter, self.batch in enumerate(self.dl):
            self._one_batch()

    def one_epoch(self, training):
        self.model.train(training)
        self.dl = self.dls.train if training else self.dls.valid
        self._one_epoch()

    @with_cbs("fit")
    def _fit(self, train, valid):
        for self.epoch in self.epochs:
            if train:
                self.one_epoch(True)
            if valid:
                torch.no_grad()(self.one_epoch)(False)

    def fit(self, n_epochs=1, train=True, valid=True):
        self.n_epochs = n_epochs
        self.epochs = range(n_epochs)
        self.opt = self.opt_func(self.model.parameters(), self.lr)
        self._fit(train, valid)

    def __getattr__(self, name):
        if name in ("predict", "get_loss", "backward", "step", "zero_grad"):
            return partial(self.callback, name)
        raise AttributeError(name)

    def callback(self, method_nm):
        run_cbs(self.cbs, method_nm, self)

    @property
    def training(self):
        return self.model.training


# %% ../nbs/09_learner.ipynb 52
class TrainLearner(Learner):
    def predict(self):
        self.preds = self.model(self.batch[0])

    def get_loss(self):
        self.loss = self.loss_func(self.preds, self.batch[1])

    def backward(self):
        self.loss.backward()

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()


## Example usage
if __name__ == "__main__":
    x, y = "image", "label"
    name = "fashion_mnist"
    import torchvision.transforms.functional as TF
    from datasets import load_dataset

    dsd = load_dataset(name)

    def transformi(b):
        b[x] = [torch.flatten(TF.to_tensor(o)) for o in b[x]]
        return b

    bs = 128
    tds = dsd.with_transform(transformi)
    dls = DataLoaders.from_dd(tds, bs, num_workers=4)
    dt = dls.train
    xb, yb = next(iter(dt))
    xb.shape, yb[:10]
    m, nh = 28 * 28, 50

    def get_model():
        return nn.Sequential(nn.Linear(m, nh), nn.ReLU(), nn.Linear(nh, 10))

    def accuracy(preds, targets):
        return (preds.argmax(dim=1) == targets).float().mean()

    model = get_model()
    learn = TrainLearner(model, dls, F.cross_entropy, lr=0.2, cbs=[DeviceCB(), ProgressCB(), MetricsCB(accuracy=accuracy)])
    learn.fit(5)
