"""Stardist 2D model modified for stacked timepoints."""
from __future__ import annotations

import warnings

import numpy as np
import tensorflow as tf
from csbdeep.internals.blocks import unet_block
from csbdeep.internals.predict import tile_iterator
from csbdeep.internals.predict import total_n_tiles
from csbdeep.utils import _raise
from csbdeep.utils import axes_dict
from csbdeep.utils.tf import IS_TF_1
from csbdeep.utils.tf import CARETensorBoard
from csbdeep.utils.tf import CARETensorBoardImage
from csbdeep.utils.tf import keras_import
from stardist.models import StarDist2D
from stardist.models.base import StarDistPadAndCropResizer
from stardist.models.base import _tf_version_at_least
from stardist.models.base import kld
from stardist.models.base import masked_loss_iou
from stardist.models.base import masked_loss_mae
from stardist.models.base import masked_loss_mse
from stardist.models.base import masked_metric_iou
from stardist.models.base import masked_metric_mae
from stardist.models.base import masked_metric_mse
from stardist.models.base import weighted_categorical_crossentropy
from stardist.utils import _is_floatarray
from tqdm import tqdm

from .config_2d import StackedTimepointsConfig2D
from .data_2d import StackedTimepointsData2D


K = keras_import("backend")
Input, Conv2D, MaxPooling2D = keras_import("layers", "Input", "Conv2D", "MaxPooling2D")
Adam = keras_import("optimizers", "Adam")
ReduceLROnPlateau, TensorBoard = keras_import(
    "callbacks", "ReduceLROnPlateau", "TensorBoard"
)
Model = keras_import("models", "Model")


class StackedTimepointsModel2D(StarDist2D):
    """Stardist model for stacked timepoints by overwriting the relevant functions."""

    def _build(self):
        """Has to be overwritten as the outputs slightly differ."""
        self.config.backbone == "unet" or _raise(NotImplementedError())
        unet_kwargs = {
            k[len("unet_") :]: v
            for (k, v) in vars(self.config).items()
            if k.startswith("unet_")
        }

        input_img = Input(self.config.net_input_shape, name="input")

        # maxpool input image to grid size
        pooled = np.array([1, 1])
        pooled_img = input_img
        while tuple(pooled) != tuple(self.config.grid):
            pool = 1 + (np.asarray(self.config.grid) > pooled)
            pooled *= pool
            for _ in range(self.config.unet_n_conv_per_depth):
                pooled_img = Conv2D(
                    self.config.unet_n_filter_base,
                    self.config.unet_kernel_size,
                    padding="same",
                    activation=self.config.unet_activation,
                )(pooled_img)
            pooled_img = MaxPooling2D(pool)(pooled_img)

        unet_base = unet_block(**unet_kwargs)(pooled_img)

        if self.config.net_conv_after_unet > 0:
            unet = Conv2D(
                self.config.net_conv_after_unet,
                self.config.unet_kernel_size,
                name="features",
                padding="same",
                activation=self.config.unet_activation,
            )(unet_base)
        else:
            unet = unet_base

        output_prob = Conv2D(
            self.config.len_t, (1, 1), name="prob", padding="same", activation="sigmoid"
        )(unet)
        output_dist = Conv2D(
            self.config.n_rays * self.config.len_t,
            (1, 1),
            name="dist",
            padding="same",
            activation="linear",
        )(unet)

        # attach extra classification head when self.n_classes is given
        if self._is_multiclass():
            if self.config.net_conv_after_unet > 0:
                unet_class = Conv2D(
                    self.config.net_conv_after_unet,
                    self.config.unet_kernel_size,
                    name="features_class",
                    padding="same",
                    activation=self.config.unet_activation,
                )(unet_base)
            else:
                unet_class = unet_base

            output_prob_class = Conv2D(
                self.config.n_classes + 1,
                (1, 1),
                name="prob_class",
                padding="same",
                activation="softmax",
            )(unet_class)
            return Model([input_img], [output_prob, output_dist, output_prob_class])
        else:
            return Model([input_img], [output_prob, output_dist])

    def prepare_for_training(self, optimizer=None):
        """Method from base class needs to be overwritten for slightly adapted loss."""
        if optimizer is None:
            optimizer = Adam(self.config.train_learning_rate)

        masked_dist_loss = {
            "mse": masked_loss_mse,
            "mae": masked_loss_mae,
            "iou": masked_loss_iou,
        }[self.config.train_dist_loss]
        prob_loss = "binary_crossentropy"

        self.num_or_size_splits = [
            self.config.n_rays for _ in range(self.config.len_t)
        ] + [1 for _ in range(self.config.len_t)]
        self.num_or_size_splits_pred = self.num_or_size_splits[: self.config.len_t]

        def split_dist_maps(dist_true_mask, dist_pred):
            return tf.split(
                dist_true_mask, num_or_size_splits=self.num_or_size_splits, axis=-1
            ), tf.split(
                dist_pred, num_or_size_splits=self.num_or_size_splits_pred, axis=-1
            )

        def dist_loss(dist_true_mask, dist_pred):
            true_splits, pred_splits = split_dist_maps(dist_true_mask, dist_pred)
            return K.mean(
                tf.stack(
                    [
                        masked_dist_loss(
                            true_splits[i + self.config.len_t],
                            reg_weight=self.config.train_background_reg,
                        )(true_splits[i], pred_splits[i])
                        for i in range(self.config.len_t)
                    ]
                )
            )

        def dist_iou_metric(dist_true_mask, dist_pred):
            true_splits, pred_splits = split_dist_maps(dist_true_mask, dist_pred)
            return K.mean(
                tf.stack(
                    [
                        masked_metric_iou(
                            true_splits[i + self.config.len_t],
                            reg_weight=0,
                        )(true_splits[i], pred_splits[i])
                        for i in range(self.config.len_t)
                    ]
                )
            )

        def relevant_mae(dist_true_mask, dist_pred):
            true_splits, pred_splits = split_dist_maps(dist_true_mask, dist_pred)
            return K.mean(
                tf.stack(
                    [
                        masked_metric_mae(
                            true_splits[i + self.config.len_t],
                        )(true_splits[i], pred_splits[i])
                        for i in range(self.config.len_t)
                    ]
                )
            )

        def relevant_mse(dist_true_mask, dist_pred):
            true_splits, pred_splits = split_dist_maps(dist_true_mask, dist_pred)
            return K.mean(
                tf.stack(
                    [
                        masked_metric_mse(
                            true_splits[i + self.config.len_t],
                        )(true_splits[i], pred_splits[i])
                        for i in range(self.config.len_t)
                    ]
                )
            )

        if self._is_multiclass():
            prob_class_loss = weighted_categorical_crossentropy(
                self.config.train_class_weights, ndim=self.config.n_dim
            )
            loss = [prob_loss, dist_loss, prob_class_loss]
        else:
            loss = [prob_loss, dist_loss]

        self.keras_model.compile(
            optimizer,
            loss=loss,
            loss_weights=list(self.config.train_loss_weights),
            metrics={
                "prob": kld,
                "dist": [relevant_mae, relevant_mse, dist_iou_metric],
            },
        )

        self.callbacks = []
        if self.basedir is not None:
            self.callbacks += self._checkpoint_callbacks()

            if self.config.train_tensorboard:
                if IS_TF_1:
                    self.callbacks.append(
                        CARETensorBoard(
                            log_dir=str(self.logdir),
                            prefix_with_timestamp=False,
                            n_images=3,
                            write_images=True,
                            prob_out=False,
                        )
                    )
                else:
                    self.callbacks.append(
                        TensorBoard(
                            log_dir=str(self.logdir / "logs"),
                            write_graph=False,
                            profile_batch=0,
                        )
                    )

        if self.config.train_reduce_lr is not None:
            rlrop_params = self.config.train_reduce_lr
            if "verbose" not in rlrop_params:
                rlrop_params["verbose"] = True
            # TF2: add as first callback to put 'lr' in the logs for TensorBoard
            self.callbacks.insert(0, ReduceLROnPlateau(**rlrop_params))

        self._model_prepared = True

    def train(
        self,
        xs,
        ys,
        validation_data,
        classes="auto",
        augmenter=None,
        seed=None,
        epochs=None,
        steps_per_epoch=None,
        workers=1,
    ):
        """This is a nearly perfect copy of StarDist2D.train."""
        if seed is not None:
            # https://keras.io/getting-started/faq/
            # #how-can-i-obtain-reproducible-results-using-keras-during-development
            np.random.seed(seed)
        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        classes = self._parse_classes_arg(classes, len(xs))

        if not self._is_multiclass() and classes is not None:
            warnings.warn("Ignoring given classes as n_classes is set to None")

        isinstance(validation_data, (list, tuple)) or _raise(ValueError())
        if self._is_multiclass() and len(validation_data) == 2:
            validation_data = tuple(validation_data) + ("auto",)
        (
            (len(validation_data) == (3 if self._is_multiclass() else 2))
            or _raise(
                ValueError(
                    f"len(validation_data) = {len(validation_data)}, "
                    f"but should be {3 if self._is_multiclass() else 2}"
                )
            )
        )

        patch_size = self.config.train_patch_size
        axes = self.config.axes.replace("C", "")
        b = (
            self.config.train_completion_crop
            if self.config.train_shape_completion
            else 0
        )
        div_by = self._axes_div_by(axes)
        [
            (p - 2 * b) % d == 0
            or _raise(
                ValueError(
                    "'train_patch_size' - 2*'train_completion_crop' must be "
                    f"divisible by {d} along axis '{a}'"
                    if self.config.train_shape_completion
                    else f"'train_patch_size' must be divisible by {d} along axis '{a}'"
                )
            )
            for p, d, a in zip(patch_size, div_by, axes)
        ]

        if not self._model_prepared:
            self.prepare_for_training()

        data_kwargs = dict(
            n_rays=self.config.n_rays,
            patch_size=self.config.train_patch_size,
            grid=self.config.grid,
            shape_completion=self.config.train_shape_completion,
            b=self.config.train_completion_crop,
            use_gpu=self.config.use_gpu,
            foreground_prob=self.config.train_foreground_only,
            n_classes=self.config.n_classes,
            sample_ind_cache=self.config.train_sample_cache,
        )

        # generate validation data and store in numpy arrays
        n_data_val = len(validation_data[0])
        classes_val = (
            self._parse_classes_arg(validation_data[2], n_data_val)
            if self._is_multiclass()
            else None
        )
        n_take = (
            self.config.train_n_val_patches
            if self.config.train_n_val_patches is not None
            else n_data_val
        )

        # CHANGED: StarDistData2D -> StackedTimepointsData2D
        _data_val = StackedTimepointsData2D(
            validation_data[0],
            validation_data[1],
            classes=classes_val,
            batch_size=n_take,
            length=1,
            **data_kwargs,
        )
        data_val = _data_val[0]

        # CHANGED: StarDistData2D -> StackedTimepointsData2D
        # expose data generator as member for general diagnostics
        self.data_train = StackedTimepointsData2D(
            xs,
            ys,
            classes=classes,
            batch_size=self.config.train_batch_size,
            augmenter=augmenter,
            length=epochs * steps_per_epoch,
            **data_kwargs,
        )

        if self.config.train_tensorboard:
            # show dist for three rays
            _n = min(3, self.config.n_rays)
            channel = axes_dict(self.config.axes)["C"]
            output_slices = [[slice(None)] * 4, [slice(None)] * 4]
            output_slices[1][1 + channel] = slice(
                0, (self.config.n_rays // _n) * _n, self.config.n_rays // _n
            )
            if self._is_multiclass():
                _n = min(3, self.config.n_classes)
                output_slices += [[slice(None)] * 4]
                output_slices[2][1 + channel] = slice(
                    1,
                    1 + (self.config.n_classes // _n) * _n,
                    self.config.n_classes // _n,
                )

            if IS_TF_1:
                for cb in self.callbacks:
                    if isinstance(cb, CARETensorBoard):
                        cb.output_slices = output_slices
                        # target image for dist includes dist_mask and thus
                        # has more channels than dist output
                        cb.output_target_shapes = [None, [None] * 4, None]
                        cb.output_target_shapes[1][1 + channel] = data_val[1][1].shape[
                            1 + channel
                        ]
            elif self.basedir is not None and not any(
                isinstance(cb, CARETensorBoardImage) for cb in self.callbacks
            ):
                self.callbacks.append(
                    CARETensorBoardImage(
                        model=self.keras_model,
                        data=data_val,
                        log_dir=str(self.logdir / "logs" / "images"),
                        n_images=3,
                        prob_out=False,
                        output_slices=output_slices,
                    )
                )

        fit = self.keras_model.fit_generator if IS_TF_1 else self.keras_model.fit
        history = fit(
            iter(self.data_train),
            validation_data=data_val,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            workers=workers,
            use_multiprocessing=workers > 1,
            callbacks=self.callbacks,
            verbose=1,
            # set validation batchsize to training batchsize (only works for tf >= 2.2)
            **(
                dict(validation_batch_size=self.config.train_batch_size)
                if _tf_version_at_least("2.2.0")
                else {}
            ),
        )
        self._training_finished()

        return history

    def _predict_setup(
        self, img, axes, normalizer, n_tiles, show_tile_progress, predict_kwargs
    ):
        """Modified version."""
        if n_tiles is None:
            n_tiles = [1] * img.ndim
        try:
            n_tiles = tuple(n_tiles)
            img.ndim == len(n_tiles) or _raise(TypeError())
        except TypeError:
            raise ValueError(
                "n_tiles must be an iterable of length %d" % img.ndim
            ) from None
        all(np.isscalar(t) and 1 <= t and int(t) == t for t in n_tiles) or _raise(
            ValueError("all values of n_tiles must be integer values >= 1")
        )

        n_tiles = tuple(map(int, n_tiles))

        axes = self._normalize_axes(img, axes)
        axes_net = self.config.axes

        _permute_axes = self._make_permute_axes(axes, axes_net)
        x = _permute_axes(img)  # x has axes_net semantics

        channel = axes_dict(axes_net)["C"]
        # MODIFIED
        self.config.n_channel_in * self.config.len_t == x.shape[channel] or _raise(
            ValueError()
        )
        axes_net_div_by = self._axes_div_by(axes_net)

        grid = tuple(self.config.grid)
        len(grid) == len(axes_net) - 1 or _raise(ValueError())
        grid_dict = dict(zip(axes_net.replace("C", ""), grid))

        normalizer = self._check_normalizer_resizer(normalizer, None)[0]
        resizer = StarDistPadAndCropResizer(grid=grid_dict)

        x = normalizer.before(x, axes_net)
        x = resizer.before(x, axes_net, axes_net_div_by)

        if not _is_floatarray(x):
            warnings.warn("Predicting on non-float input... ( forgot to normalize? )")

        def predict_direct(x):
            ys = self.keras_model.predict(x[np.newaxis], **predict_kwargs)
            return tuple(y[0] for y in ys)

        def tiling_setup():
            np.prod(n_tiles) > 1 or _raise(
                ValueError("tiling setup for n_tiles = (1, 1, 1)")
            )
            tiling_axes = axes_net.replace("C", "")  # axes eligible for tiling
            x_tiling_axis = tuple(
                axes_dict(axes_net)[a] for a in tiling_axes
            )  # numerical axis ids for x
            axes_net_tile_overlaps = self._axes_tile_overlap(axes_net)
            # hack: permute tiling axis in the same way as img -> x was permuted
            _n_tiles = _permute_axes(np.empty(n_tiles, bool)).shape
            (
                all(_n_tiles[i] == 1 for i in range(x.ndim) if i not in x_tiling_axis)
                or _raise(
                    ValueError(
                        "entry of n_tiles > 1 only allowed for axes '%s'" % tiling_axes
                    )
                )
            )

            sh = [s // grid_dict.get(a, 1) for a, s in zip(axes_net, x.shape)]
            sh[channel] = None

            def create_empty_output(n_channel, dtype=np.float32):
                sh[channel] = n_channel
                return np.empty(sh, dtype)

            if callable(show_tile_progress):
                progress, _show_tile_progress = show_tile_progress, True
            else:
                progress, _show_tile_progress = tqdm, show_tile_progress

            n_block_overlaps = [
                int(np.ceil(overlap / blocksize))
                for overlap, blocksize in zip(axes_net_tile_overlaps, axes_net_div_by)
            ]

            num_tiles_used = total_n_tiles(
                x,
                _n_tiles,
                block_sizes=axes_net_div_by,
                n_block_overlaps=n_block_overlaps,
            )

            tile_generator = progress(
                tile_iterator(
                    x,
                    _n_tiles,
                    block_sizes=axes_net_div_by,
                    n_block_overlaps=n_block_overlaps,
                ),
                disable=(not _show_tile_progress),
                total=num_tiles_used,
            )

            return tile_generator, tuple(sh), create_empty_output

        return (
            x,
            axes,
            axes_net,
            axes_net_div_by,
            _permute_axes,
            resizer,
            n_tiles,
            grid,
            grid_dict,
            channel,
            predict_direct,
            tiling_setup,
        )

    def _predict_generator(
        self,
        img,
        axes=None,
        normalizer=None,
        n_tiles=None,
        show_tile_progress=True,
        **predict_kwargs,
    ):
        """Modified version of the version from StarDist2D."""
        (
            x,
            axes,
            axes_net,
            axes_net_div_by,
            _permute_axes,
            resizer,
            n_tiles,
            grid,
            grid_dict,
            channel,
            predict_direct,
            tiling_setup,
        ) = self._predict_setup(
            img, axes, normalizer, n_tiles, show_tile_progress, predict_kwargs
        )

        if np.prod(n_tiles) > 1:
            tile_generator, output_shape, create_empty_output = tiling_setup()

            # MODIFIED
            prob = create_empty_output(self.config.len_t)
            dist = create_empty_output(self.config.n_rays * self.config.len_t)
            if self._is_multiclass():
                prob_class = create_empty_output(self.config.n_classes + 1)
                result = (prob, dist, prob_class)
            else:
                result = (prob, dist)

            for tile, s_src, s_dst in tile_generator:
                # predict_direct -> prob, dist, [prob_class if multi_class]
                result_tile = predict_direct(tile)
                # account for grid
                s_src = [
                    slice(s.start // grid_dict.get(a, 1), s.stop // grid_dict.get(a, 1))
                    for s, a in zip(s_src, axes_net)
                ]
                s_dst = [
                    slice(s.start // grid_dict.get(a, 1), s.stop // grid_dict.get(a, 1))
                    for s, a in zip(s_dst, axes_net)
                ]
                # prob and dist have different channel dimensionality than image x
                s_src[channel] = slice(None)
                s_dst[channel] = slice(None)
                s_src, s_dst = tuple(s_src), tuple(s_dst)
                # print(s_src,s_dst)
                for part, part_tile in zip(result, result_tile):
                    part[s_dst] = part_tile[s_src]
                yield  # yield None after each processed tile
        else:
            # predict_direct -> prob, dist, [prob_class if multi_class]
            result = predict_direct(x)

        result = [resizer.after(part, axes_net) for part in result]

        # result = (prob, dist) for legacy or (prob, dist, prob_class) for multiclass

        # prob
        # result[0] = np.take(result[0], 0, axis=channel)
        # dist
        result[1] = np.maximum(
            1e-3, result[1]
        )  # avoid small dist values to prevent problems with Qhull
        result[1] = np.moveaxis(result[1], channel, -1)

        if self._is_multiclass():
            # prob_class
            result[2] = np.moveaxis(result[2], channel, -1)

        # last "yield" is the actual output that would
        # have been "return"ed if this was a regular function
        yield tuple(result)

    def predict_tyx(self, x):
        """Prepare input image of shape TYXC to YXC for internal representation."""
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)
        x = np.concatenate([x[i] for i in range(self.config.len_t)], axis=-1)
        prob, dists = self.predict(x)

        prob = np.transpose(prob, (2, 0, 1))

        dists = np.stack(np.split(dists, self.config.len_t, axis=-1), axis=0)

        return prob, dists

    @property
    def _config_class(self):
        """Needed method for the config class to use."""
        return StackedTimepointsConfig2D
