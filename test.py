"""Test rewritten modules."""
import os
from glob import glob

import numpy as np
from tifffile import imread
from tifffile import imwrite

from stardist_stacked_timepoints.config_2d import StackedTimepointsConfig2D
from stardist_stacked_timepoints.data_2d import StackedTimepointsData2D
from stardist_stacked_timepoints.model_2d import StackedTimepointsModel2D


# from tifffile import imwrite


path = (
    "/mnt/d/training-data/kazumasa-takeuchi/"
    "2022-10-20-kazumasa-takeuchi-special/kazumasa-for-simon"
)

Y_files = glob(os.path.join(path, "*_lbl.tif"))
X_files = [f.replace("_lbl", "") for f in Y_files]

Y = list(map(imread, Y_files))
X = list(map(imread, X_files))
print(len(X))
print(len(Y))
print(X[6].shape)

gen = StackedTimepointsData2D(X, Y, 2, 16, 100, patch_size=(64, 64))
gen[0]
val_ind = 5
val = StackedTimepointsData2D(X[val_ind:], Y[val_ind:], 20, 8, 100, patch_size=(32, 32))
val[0]

conf = StackedTimepointsConfig2D(
    n_rays=8,
    grid=(1, 1),
    len_t=3,
    n_channel_in=1,
    train_patch_size=(32, 32),
    train_batch_size=2,
    train_epochs=100,
)
model = StackedTimepointsModel2D(conf, name="model", basedir=path)


def random_fliprot_2d(img, mask):
    """Flip or rotate arrays of shape (T, Y, X, C) in Y and X direction."""
    axes = (1, 2)
    perm = tuple(np.random.permutation(axes))
    img = img.transpose((0,) + perm + (3,))
    mask = mask.transpose((0,) + perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img):
    """Add random intensity for changing it."""
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


def augmenter_2d(x, y):
    """Augmentation of a single input/label image pair.

    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot_2d(x, y)
    # here some random distortions in the different timepoints could be added
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02 * np.random.uniform(0, 1)
    x = x + sig * np.random.normal(0, 1, x.shape)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


X = [np.expand_dims(x, axis=-1) for x in X]
fov = np.array(model._axes_tile_overlap("YX"))
print(fov)

history = model.train(
    X[:val_ind], Y[:val_ind], (X[val_ind:], Y[val_ind:]), augmenter=augmenter_2d
)

out = model.predict_tyx(X[0])

imwrite(os.path.join(path, "pred_dists.tif"), out[1])
