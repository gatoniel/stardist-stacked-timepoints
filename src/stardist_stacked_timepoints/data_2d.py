"""Data generator for 2d time stacks based on stardist's data generators."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom
from skimage.segmentation import clear_border
from stardist.geometry import star_dist
from stardist.models.model2d import StarDistData2D
from stardist.sample_patches import sample_patches as sd_sample_patches
from stardist.utils import edt_prob
from stardist.utils import mask_to_categorical

from .data_base import StackedTimepointsDataBase
from .sample_patches import sample_patches
from .utils import bordering_gaussian_weights
from .utils import bordering_pixels


class StackedTimepointsData2D(StackedTimepointsDataBase):
    """This class expects arrays of shape (size, T, Y, X, channels)."""

    def __init__(
        self,
        xs,
        ys,
        batch_size,
        n_rays,
        length,
        n_classes=None,
        classes=None,
        patch_size=(256, 256),
        b=32,
        grid=(1, 1),
        shape_completion=False,
        augmenter=None,
        foreground_prob=0,
        **kwargs,
    ):
        """Initialize with arrays of shape (size, T, Y, X, channels)."""
        super().__init__(
            xs=xs,
            ys=ys,
            n_rays=n_rays,
            grid=grid,
            n_classes=n_classes,
            classes=classes,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            augmenter=augmenter,
            foreground_prob=foreground_prob,
            **kwargs,
        )

        self.shape_completion = bool(shape_completion)
        if self.shape_completion and b > 0:
            self.b = slice(None), slice(b, -b), slice(b, -b)
        else:
            self.b = slice(None), slice(None), slice(None)

        self.sd_mode = "opencl" if self.use_gpu else "cpp"

    def __getitem__(self, i):
        """Return batch i as numpy array."""
        idx = self.batch(i)
        arrays = [
            sample_patches(
                (self.ys[k],) + self.channels_as_tuple(self.xs[k]),
                patch_size=self.patch_size,
                n_samples=1,
                valid_inds=self.get_valid_inds(k),
            )
            for k in idx
        ]

        if self.n_channel is None:
            xs, ys = list(zip(*[(x[0][self.b], y[0]) for y, x in arrays]))
        else:
            xs, ys = list(
                zip(
                    *[
                        (np.stack([_x[0] for _x in x], axis=-1)[self.b], y[0])
                        for y, *x in arrays
                    ]
                )
            )

        xs, ys = tuple(zip(*tuple(self.augmenter(_x, _y) for _x, _y in zip(xs, ys))))

        prob = np.stack(
            [
                np.stack(
                    [
                        edt_prob(lbl[(i,) + self.b[1:]][self.ss_grid[1:3]])
                        for i in range(self.len_t)
                    ],
                    axis=-1,
                )
                for lbl in ys
            ]
        )
        # prob = np.stack([edt_prob(lbl[self.b]) for lbl in ys])
        # prob = prob[self.ss_grid]

        if self.shape_completion:
            ys_cleared = [clear_border(lbl) for lbl in ys]
            _dist = np.stack(
                [
                    np.concat(
                        [
                            star_dist(lbl[i], self.n_rays, mode=self.sd_mode)[
                                self.b[1:] + (slice(None),)
                            ]
                            for i in range(self.len_t)
                        ],
                        axis=-1,
                    )
                    for lbl in ys_cleared
                ]
            )
            dist = _dist[self.ss_grid]
            dist_mask = np.stack(
                [
                    np.stack(
                        [
                            edt_prob(lbl[tuple(i) + self.b[1:]][self.ss_grid[1:3]])
                            for i in range(self.len_t)
                        ],
                        axis=-1,
                    )
                    for lbl in ys_cleared
                ]
            )
        else:
            # directly subsample with grid
            dist = np.stack(
                [
                    np.concatenate(
                        [
                            star_dist(
                                lbl[i], self.n_rays, mode=self.sd_mode, grid=self.grid
                            )
                            for i in range(self.len_t)
                        ],
                        axis=-1,
                    )
                    for lbl in ys
                ]
            )
            dist_mask = prob

        if xs[0].ndim == 3:
            xs = [np.expand_dims(x, axis=-1) for x in xs]
        xs = np.stack(
            [np.concatenate([x[i] for i in range(self.len_t)], axis=-1) for x in xs]
        )

        # subsample wth given grid
        # dist_mask = dist_mask[self.ss_grid]
        # prob      = prob[self.ss_grid]

        # append dist_mask to dist as additional channel
        # dist_and_mask = np.concatenate([dist,dist_mask],axis=-1)
        # faster than concatenate
        dist_and_mask = np.empty(
            dist.shape[:-1] + (self.len_t * (self.n_rays + 1),), np.float32
        )
        dist_and_mask[..., : -self.len_t] = dist
        dist_and_mask[..., -self.len_t :] = dist_mask

        return [xs], [prob, dist_and_mask]


class OptimizedStarDistData2D(StarDistData2D):
    """Overwrite __getitem__ function to use different prob and weights."""

    def __getitem__(self, i):
        """Return batch i as numpy array."""
        idx = self.batch(i)
        arrays = [
            sd_sample_patches(
                (self.Y[k],) + self.channels_as_tuple(self.X[k]),
                patch_size=self.patch_size,
                n_samples=1,
                valid_inds=self.get_valid_inds(k),
            )
            for k in idx
        ]

        if self.n_channel is None:
            xs, ys = list(zip(*[(x[0][self.b], y[0]) for y, x in arrays]))
        else:
            xs, ys = list(
                zip(
                    *[
                        (np.stack([_x[0] for _x in x], axis=-1)[self.b], y[0])
                        for y, *x in arrays
                    ]
                )
            )

        xs, ys = tuple(zip(*tuple(self.augmenter(_x, _y) for _x, _y in zip(xs, ys))))

        _prob = np.stack([edt_prob(lbl[self.b][self.ss_grid[1:3]]) for lbl in ys])
        bordering_map = np.stack(
            [bordering_pixels(lbl[self.b][self.ss_grid[1:3]]) for lbl in ys]
        )
        prob = np.clip(_prob - bordering_map, 0, 1)
        # prob = np.stack([edt_prob(lbl[self.b]) for lbl in ys])
        # prob = prob[self.ss_grid]

        if self.shape_completion:
            ys_cleared = [clear_border(lbl) for lbl in ys]
            _dist = np.stack(
                [
                    star_dist(lbl, self.n_rays, mode=self.sd_mode)[
                        self.b + (slice(None),)
                    ]
                    for lbl in ys_cleared
                ]
            )
            dist = _dist[self.ss_grid]
            dist_mask_prob = np.stack(
                [edt_prob(lbl[self.b][self.ss_grid[1:3]]) for lbl in ys_cleared]
            )
            dist_mask_bordering = [
                bordering_pixels(lbl[self.b][self.ss_grid[1:3]]) for lbl in ys_cleared
            ]
            dist_mask_edt = np.stack(
                [
                    bordering_gaussian_weights(mask, lbl, sigma=2)
                    for mask, lbl in zip(dist_mask_bordering, ys_cleared)
                ]
            )
            dist_mask = dist_mask_prob + dist_mask_edt

        else:
            # directly subsample with grid
            dist = np.stack(
                [
                    star_dist(lbl, self.n_rays, mode=self.sd_mode, grid=self.grid)
                    for lbl in ys
                ]
            )
            dist_mask_bordering = [
                bordering_pixels(lbl[self.b][self.ss_grid[1:3]]) for lbl in ys
            ]
            dist_mask_edt = np.stack(
                [
                    bordering_gaussian_weights(mask, lbl, sigma=2)
                    for mask, lbl in zip(dist_mask_bordering, ys)
                ]
            )
            dist_mask = _prob + dist_mask_edt

        xs = np.stack(xs)
        if xs.ndim == 3:  # input image has no channel axis
            xs = np.expand_dims(xs, -1)
        prob = np.expand_dims(prob, -1)
        dist_mask = np.expand_dims(dist_mask, -1)

        # subsample wth given grid
        # dist_mask = dist_mask[self.ss_grid]
        # prob      = prob[self.ss_grid]

        # append dist_mask to dist as additional channel
        # dist_and_mask = np.concatenate([dist,dist_mask],axis=-1)
        # faster than concatenate
        dist_and_mask = np.empty(dist.shape[:-1] + (self.n_rays + 1,), np.float32)
        dist_and_mask[..., :-1] = dist
        dist_and_mask[..., -1:] = dist_mask

        if self.n_classes is None:
            return [xs], [prob, dist_and_mask]
        else:
            prob_class = np.stack(
                tuple(
                    (
                        mask_to_categorical(y, self.n_classes, self.classes[k])
                        for y, k in zip(ys, idx)
                    )
                )
            )

            # TODO: investigate downsampling via simple indexing vs. using 'zoom'
            # prob_class = prob_class[self.ss_grid]
            # 'zoom' might lead to better registered maps (especially if upscaled later)
            prob_class = zoom(
                prob_class, (1,) + tuple(1 / g for g in self.grid) + (1,), order=0
            )

            return [xs], [prob, dist_and_mask, prob_class]
