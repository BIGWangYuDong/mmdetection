_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='FFTFilter', shape='cycle', pass_type='low', radius=16),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

model = dict(
    rpn_head=dict(
        loss_cls=dict(loss_weight=0.5), loss_bbox=dict(loss_weight=0.5)),
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(loss_weight=0.5), loss_bbox=dict(loss_weight=0.5))))
