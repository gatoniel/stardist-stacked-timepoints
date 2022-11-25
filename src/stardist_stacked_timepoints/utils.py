"""Functions to calculate optimized prob maps and distance weights."""
import numba
import numpy as np
import numpy.typing as npt
from edt import edt
from numba import njit
from scipy.ndimage import binary_dilation
from scipy.ndimage import generate_binary_structure


def bordering_pixels_arbitrary(lbl):
    """Slowly calculate the pixels of objects that touch other objects in 2d and 3d."""
    struct = generate_binary_structure(lbl.ndim, lbl.ndim)
    mask = lbl > 0
    lbl_ids = np.unique(lbl[mask])
    expanded = np.zeros(lbl.shape, dtype=bool)
    for lbl_id in lbl_ids:
        mask_ = lbl == lbl_id
        dilation = binary_dilation(mask_, struct)
        expanded_ = np.logical_xor(mask_, dilation)
        expanded = np.logical_or(expanded, expanded_)
    return np.logical_and(mask, expanded)


@njit
def determine_neighbor_2d(
    y: int,
    off_y: int,
    x: int,
    off_x: int,
    lbl: npt.NDArray[int],
    mask: npt.NDArray[bool],
    bordering: npt.NDArray[bool],
) -> None:
    """Utility function that is called several times in the below function."""
    y_ = y + off_y
    x_ = x + off_x
    if mask[y_, x_] and lbl[y, x] != lbl[y_, x_]:
        bordering[y, x] = True
        bordering[y_, x_] = True


@njit
def bordering_pixels_2d(lbl: npt.NDArray[int]) -> npt.NDArray[bool]:
    """Calculate the pixels of objects that touch other objects."""
    bordering = np.zeros(lbl.shape, dtype=numba.types.bool_)
    mask = lbl > 0
    for y in range(lbl.shape[0] - 1):
        for x in range(1, lbl.shape[1] - 1):
            if mask[y, x]:
                for (off_y, off_x) in [(1, -1), (0, 1), (1, 1), (1, 0)]:
                    determine_neighbor_2d(y, off_y, x, off_x, lbl, mask, bordering)
        x = 0
        if mask[y, x]:
            for (off_y, off_x) in [(0, 1), (1, 1), (1, 0)]:
                determine_neighbor_2d(y, off_y, x, off_x, lbl, mask, bordering)

        x = lbl.shape[1] - 1
        if mask[y, x]:
            off_y = 1
            off_x = 0
            determine_neighbor_2d(y, off_y, x, off_x, lbl, mask, bordering)

    y = lbl.shape[0] - 1
    for x in range(0, lbl.shape[1] - 1):
        if mask[y, x]:
            off_y = 0
            off_x = 1
            determine_neighbor_2d(y, off_y, x, off_x, lbl, mask, bordering)

    return bordering


def bordering_gaussian_weights(border_pixels, lbl, sigma=2):
    """Gaussian of edt from border_pixels only for pixels with lbl > 0."""
    bordering_edt = edt(np.logical_not(border_pixels))
    bordering_weight = np.zeros_like(lbl, dtype=float)
    _mask = lbl > 0
    bordering_weight[_mask] = np.exp(-np.square(bordering_edt[_mask]) / 2 / sigma**2)

    return bordering_weight
