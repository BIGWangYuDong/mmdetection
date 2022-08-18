# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
from mmengine.fileio import FileClient
from mmengine.visualization import Visualizer


def featmap_vis(feature_maps, batch_data_samples, results_list, name):
    visualizer = Visualizer.get_current_instance()
    assert len(feature_maps) == len(name)
    for i in range(len(feature_maps)):
        if i < 1:
            continue
        featmap = feature_maps[i][0].detach()
        assert featmap.ndim == 3
        # load_img
        img_path = batch_data_samples[0].metainfo['img_path']
        file_client_args: dict = dict(backend='disk')
        file_client_args = file_client_args.copy()
        file_client = FileClient(**file_client_args)
        img_bytes = file_client.get(img_path)
        image = mmcv.imfrombytes(
            img_bytes, flag='color', backend='cv2', channel_order='rgb')
        if image.shape[0] > 2000:
            fake_img_shape = (int(image.shape[0]/2), int(image.shape[1]/2))
            feat_img = visualizer.draw_featmap(
                featmap,
                image,
                topk=64,
                resize_shape=fake_img_shape,
                channel_reduction='squeeze_mean',
                alpha=0.35)
            feat_img = mmcv.imresize(feat_img, (image.shape[1], image.shape[0]))
        else:
            # feat_map
            feat_img = visualizer.draw_featmap(
                featmap,
                image,
                topk=64,
                channel_reduction='squeeze_mean',
                alpha=0.35)

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
        # show gt
        visualizer.add_datasample(
            f'image_gt_{name[i]}_{osp.split(img_path)[-1][:-4]}_',
            image,
            gt_sample=batch_data_samples[0].cpu(),
            pred_sample=None,
            palette=[(210, 105, 30) for _ in range(4)])
