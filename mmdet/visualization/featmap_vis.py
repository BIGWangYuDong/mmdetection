# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
from mmengine.visualization import Visualizer


def featmap_vis(feature_maps, batch_data_samples, results_list, name):
    visualizer = Visualizer.get_current_instance()
    assert len(feature_maps) == len(name)
    for i in range(len(feature_maps)):
        featmap = feature_maps[i][0].detach()
        assert featmap.ndim == 3
        # load_img
        img_path = batch_data_samples[0].metainfo['img_path']
        file_client_args: dict = dict(backend='disk')
        file_client_args = file_client_args.copy()
        file_client = mmcv.FileClient(**file_client_args)
        img_bytes = file_client.get(img_path)
        image = mmcv.imfrombytes(
            img_bytes, flag='color', backend='cv2', channel_order='rgb')
        if image.shape[0] > 2000:
            continue
        # feat_map
        feat_img = visualizer.draw_featmap(
            featmap,
            image,
            topk=64,
            channel_reduction='squeeze_mean',
            alpha=0.2)
        del featmap, image
        # visualize image with feature map and add gt or pred bbox
        visualizer.add_datasample(
            f'pred_{name[i]}_{osp.split(img_path)[-1][:-4]}_',
            feat_img,
            gt_sample=None,
            pred_sample=results_list[0])
        visualizer.add_datasample(
            f'gt_{name[i]}_{osp.split(img_path)[-1][:-4]}_',
            feat_img,
            gt_sample=batch_data_samples[0].cpu(),
            pred_sample=None)
