"""Functions to directly predict instances on timeseries data."""
from __future__ import annotations

from typing import TypeVar

import numpy as np
import numpy.typing as npt


T = TypeVar("T", bound=np.generic)
S = TypeVar("S", bound=np.generic)


def timeseries_to_batch(x: npt.NDArray[T], len_t: int = 3) -> list[npt.NDArray[T]]:
    """Turn the array of shape (T, Y, X, c) into subarrays of shape (len_t, Y, X, c)."""
    subviews = np.moveaxis(
        np.lib.stride_tricks.sliding_window_view(x, len_t, axis=0), -1, 1
    )

    list_subviews = np.split(subviews, subviews.shape[0], axis=0)
    return [np.take(subview, 0, axis=0) for subview in list_subviews]


def average_over_window(windows):
    """Calculate average over sliding windows."""
    len_w = windows.shape[1]
    empty = np.full(windows[0, :1, ...].shape, fill_value=np.nan)

    total_num = len_w - 1
    arrays = []
    for i in range(len_w):
        before = np.repeat(empty, i, axis=0)
        after = np.repeat(empty, total_num - i, axis=0)
        arrays.append(np.concatenate((before, windows[:, i, ...], after), axis=0))
    return np.nanmean(np.stack(arrays, axis=-1), axis=-1)


# def averaged_network_output(x: npt.NDArray[T], model) -> tuple[npt.NDArray[S]]:
#     probs_, dists_ = tuple(zip(*tuple(model.predict_tyx_array(x))))
#
#     probs_ = np.array(probs_)
#     dists_ = np.array(dists_)
#
#     empty_prob = np.empty_like(probs_[0, 0, ...])
#     empty_dist = np.empty_like(dists_[0, 0, ...])
