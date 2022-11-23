# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class PODDataset(CocoDataset):

    CLASSES = ('box', 'fork', 'label', 'pod')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)]
