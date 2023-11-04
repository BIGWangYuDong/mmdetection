from mmdet.datasets import CocoDataset

from mmdet.registry import DATASETS


URPC_METAINFO = {
    'classes': ('holothurian', 'echinus', 'starfish', 'scallop', 'waterweeds'),
    'palette': [(235, 211, 70), (106, 90, 205), (160, 32, 240), (176, 23, 31), (0, 0, 0)]
}


@DATASETS.register_module()
class URPCCocoDataset(CocoDataset):
    """Underwater Robot Professional Contest dataset `URPC.
    <https://arxiv.org/abs/2106.05681>`_
    
    With waterweeds
    """
    METAINFO = URPC_METAINFO
