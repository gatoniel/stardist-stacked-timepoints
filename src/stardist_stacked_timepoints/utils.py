"""Functions to calculate optimized prob maps and distance weights."""
import numpy as np
from edt import edt


def bordering_pixels(lbl):
    """Calculate the pixels of objects that touch other objects."""
    ids = np.unique(lbl[lbl > 0])
    edts = np.empty((lbl.shape) + (len(ids),))
    edts = np.stack([edt(np.logical_not(lbl == i)) for i in ids], axis=-1)
    sorted_edts = np.sort(edts, axis=-1)
    return np.logical_and(
        sorted_edts[..., 0] == 0,  # pixels that are part of some objects
        sorted_edts[..., 1] < 2,  # pixels that are also bordering at least one object
    )


def bordering_gaussian_weights(border_pixels, lbl, sigma=2):
    """Gaussian of edt from border_pixels only for pixels with lbl > 0."""
    bordering_edt = edt(np.logical_not(border_pixels))
    bordering_weight = np.zeros_like(lbl, dtype=float)
    _mask = lbl > 0
    bordering_weight[_mask] = np.exp(-np.square(bordering_edt[_mask]) / 2 / sigma**2)

    return bordering_weight
