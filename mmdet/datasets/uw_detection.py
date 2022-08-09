# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class UWCocoDataset(CocoDataset):
    METAINFO = {
        'CLASSES': ('holothurian', 'echinus', 'scallop', 'starfish'),
        # PALETTE is a list of color tuples, which is used for visualization.
        'PALETTE': [(255, 215, 60), (106, 90, 205), (160, 32, 240),
                    (176, 23, 31)]
    }
