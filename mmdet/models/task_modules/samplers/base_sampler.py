# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
from mmengine.structures import InstanceData

from .sampling_result import SamplingResult


class BaseSampler(metaclass=ABCMeta):
    """Base class of samplers.

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_up (int): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """

    def __init__(self,
                 num: int,
                 pos_fraction: float,
                 neg_pos_ub: int = -1,
                 add_gt_as_proposals: bool = True,
                 **kwargs) -> None:
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, pred_instances: InstanceData, num_expected: int,
                    **kwargs):
        """Sample positive samples."""
        pass

    @abstractmethod
    def _sample_neg(self, pred_instances: InstanceData, num_expected: int,
                    **kwargs):
        """Sample negative samples."""
        pass

    def sample(self, pred_instances: InstanceData, gt_instances: InstanceData,
               **kwargs) -> SamplingResult:
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Assigning results.
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).

        Returns:
            :obj:`SamplingResult`: Sampling result.

        Example:
            >>> from mmengine.structures import InstanceData
            >>> from mmdet.models.task_modules.samplers import RandomSampler,
            >>> from mmdet.models.task_modules.assigners import AssignResult
            >>> from mmdet.models.task_modules.samplers.
            ... sampling_result import ensure_rng, random_boxes
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> pred_instances = InstanceData()
            >>> pred_instances.priors = random_boxes(assign_result.num_preds,
            ...                                      rng=rng)
            >>> gt_instances = InstanceData()
            >>> gt_instances.bboxes = random_boxes(assign_result.num_gts,
            ...                                    rng=rng)
            >>> gt_instances.labels = torch.randint(
            ...     0, 5, (assign_result.num_gts,), dtype=torch.long)
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, pred_instances, gt_instances)
        """
        # get and delete metainfo in the pred_instance
        meta_info = pred_instances.metainfo
        pred_instances = pred_instances.new(metainfo={})

        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors
        gt_labels = gt_instances.labels
        if len(priors.shape) < 2:
            priors = priors[None, :]

        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            # When `gt_bboxes` and `priors` are all box type, convert
            # `gt_bboxes` type to `priors` type.

            num_gts = len(gt_instances)

            add_gt_instances = InstanceData()
            add_gt_instances.priors = gt_bboxes
            add_gt_instances.labels = gt_labels
            add_gt_instances.pos_gt_bboxes = gt_bboxes

            add_gt_instances.gt_flags = priors.new_ones((num_gts, ),
                                                        dtype=torch.bool)

            add_gt_instances.gt_inds = torch.arange(
                1,
                len(gt_labels) + 1,
                dtype=torch.long,
                device=gt_labels.device)
            add_gt_instances.max_overlaps = priors.new_ones((num_gts, ),
                                                            dtype=torch.bool)

            add_gt_instances.neg_inds = priors.new_zeros((num_gts, ),
                                                         dtype=torch.bool)
            add_gt_instances.pos_inds = priors.new_ones((num_gts, ),
                                                        dtype=torch.bool)
            for k, v in pred_instances.items():
                if k not in add_gt_instances:
                    shape = list(v.shape)
                    shape[0] = num_gts
                    add_gt_instances[k] = priors.new_ones(shape)

            pred_instances = pred_instances.cat(
                [add_gt_instances,
                 pred_instances.new(metainfo={})])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(pred_instances,
                                                num_expected_pos, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()

        num_bboxes = len(pred_instances)
        pos_inds_flag = priors.new_zeros((num_bboxes, ), dtype=torch.bool)
        pos_inds_flag[pos_inds] = True
        pred_instances.pos_inds = pos_inds_flag

        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(pred_instances,
                                                num_expected_neg, **kwargs)
        neg_inds = neg_inds.unique()

        neg_inds_flag = priors.new_zeros((num_bboxes, ), dtype=torch.bool)
        neg_inds_flag[neg_inds] = True
        pred_instances.neg_inds = neg_inds_flag

        # update necessary metainfos
        num_pos = max(pos_inds.numel(), 1)
        num_neg = max(neg_inds.numel(), 1)
        avg_factor = num_pos + num_neg
        meta_info.update(
            avg_factor=avg_factor, num_pos=num_pos, num_neg=num_neg)
        pred_instances.set_metainfo(metainfo=meta_info)

        return pred_instances
