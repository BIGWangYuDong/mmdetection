_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/uwdet_coco_style_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=4)))

# max keep 2 checkpoint
checkpoint_config = dict(max_keep_ckpts=2)
