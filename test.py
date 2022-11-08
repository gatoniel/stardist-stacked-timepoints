"""Test rewritten modules."""
import os
from glob import glob

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

history = model.train(X[:val_ind], Y[:val_ind], (X[val_ind:], Y[val_ind:]))

out = model.predict_tyx(X[0])

imwrite(os.path.join(path, "pred_dists.tif"), out[1])
