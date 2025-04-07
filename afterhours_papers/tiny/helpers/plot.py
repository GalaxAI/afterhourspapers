import matplotlib.pyplot as plt
import numpy as np


def show_image(im, ax=None, figsize=None, title=None, noframe=True, **kwargs):
    """Show a PIL, PyTorch, or NumPy image on `ax`."""
    if hasattr(im, "cpu") and hasattr(im, "permute") and hasattr(im, "detach"):  # PyTorch tensor
        im = im.detach().cpu()
        if len(im.shape) == 3 and im.shape[0] < 5:  # CHW to HWC for <5 channels
            im = im.permute(1, 2, 0)
    elif not isinstance(im, np.ndarray):  # PIL Image or similar
        im = np.array(im)
    if im.shape[-1] == 1:  # Grayscale (H,W,1) -> (H,W)
        im = im[..., 0]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, **kwargs)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if noframe:
        ax.axis("off")
    return ax
