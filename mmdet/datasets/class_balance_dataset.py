# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
from mmengine.dataset import ClassBalancedDataset as CBD

from mmdet.registry import DATASETS


@DATASETS.register_module()
class ClassBalancedDataset(CBD):
    """A wrapper of class balanced dataset.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in the `paper <https://arxiv.org/abs/1908.03195>`_,
    in each epoch, an image may appear multiple times based on its
    "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :meth:`get_cat_ids` to support
    ClassBalancedDataset.

    The repeat factor is computed as followed.

    1. For each category c, compute the fraction # of images
       that contain it: :math:`f(c)`
    2. For each category c, compute the category-level repeat factor:
       :math:`r(c) = max(1, sqrt(t/f(c)))`
    3. For each image I, compute the image-level repeat factor:
       :math:`r(I) = max_{c in I} r(c)`

    Note:
        ``ClassBalancedDataset`` should not inherit from ``BaseDataset``
        since ``get_subset`` and ``get_subset_`` could  produce ambiguous
        meaning sub-dataset which conflicts with original dataset. If you
        want to use a sub-dataset of ``ClassBalancedDataset``, you should set
        ``indices`` arguments for wrapped dataset which inherit from
        ``BaseDataset``.

    Args:
        dataset (BaseDataset or dict): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c >= oversample_thr``, there is
            no oversampling. For categories with ``f_c < oversample_thr``, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
        lazy_init (bool, optional): whether to load annotation during
            instantiation. Defaults to False
    """

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        # Get repeat factors for each image.
        repeat_factors = self._get_repeat_factors(self.dataset,
                                                  self.oversample_thr)
        # Repeat dataset's indices according to repeat_factors. For example,
        # if `repeat_factors = [1, 2, 3]`, and the `len(dataset) == 3`,
        # the repeated indices will be [1, 2, 2, 3, 3, 3].
        repeat_indices = []
        for dataset_index, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_index] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(self.dataset, 'flag'):
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)

        self._fully_initialized = True
