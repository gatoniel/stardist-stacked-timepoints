"""Test predictions."""
import os
from glob import glob

from tifffile import imread

from stardist_stacked_timepoints.model_2d import StackedTimepointsModel2D


path = (
    "/mnt/d/training-data/kazumasa-takeuchi/"
    "2022-10-20-kazumasa-takeuchi-special/kazumasa-for-simon"
)

Y_files = glob(os.path.join(path, "*_lbl.tif"))
X_files = [f.replace("_lbl", "") for f in Y_files]

Y = list(map(imread, Y_files))
X = list(map(imread, X_files))

model = StackedTimepointsModel2D(None, name="model", basedir=path)
