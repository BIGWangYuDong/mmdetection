_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/uwdet_coco_style_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=4)))

val_dataloader = dict(
    dataset=dict(
        ann_file='annotation_json/vis.json',
        data_prefix=dict(img='0_uwdet_RAW/')))
test_dataloader = val_dataloader
visualizer = dict(
    save_dir='work_dirs/uw_vis/faster_1x/0_raw')
# max keep 2 checkpoint
checkpoint_config = dict(max_keep_ckpts=2)
