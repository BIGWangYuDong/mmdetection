# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from numpy import random

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class FFTFilter(BaseTransform):

    def __init__(self, shape='cycle', pass_type='low', radius=16, prob=None):
        # TODO: support rhombus
        assert shape in ['cycle', 'square']
        assert pass_type in ['low', 'high', 'random', 'none']
        self.shape = shape
        self.pass_type = pass_type
        self.radius = radius
        if pass_type == 'random':
            assert prob is not None
        self.prob = prob

    @cache_randomness
    def _random_prob(self) -> float:
        return random.uniform(0, 1)

    def _hard_cycle_mask(self, results, radius):
        height, width = results['img_shape']
        x_c, y_c = width // 2, height // 2

        assert x_c >= radius and y_c >= radius

        y, x = np.ogrid[:2 * radius, :2 * radius]
        y = y[::-1]

        cycle_mask = ((x - radius)**2 + (y - radius)**2) <= radius**2

        mask = np.zeros((height, width), dtype=np.bool)
        mask[y_c - radius:y_c + radius, x_c - radius:x_c + radius] = cycle_mask
        return mask

    def _square_mask(self, results, radius):
        height, width = results['img_shape']
        x_c, y_c = width // 2, height // 2
        assert x_c >= radius and y_c >= radius

        mask = np.zeros((height, width), dtype=np.bool)
        mask[y_c - radius:y_c + radius, x_c - radius:x_c + radius] = 1
        return mask

    def transform(self, results: dict):
        if isinstance(self.radius, int) or isinstance(self.radius, float):
            radius = self.radius
        elif isinstance(self.radius, list):
            assert len(self.radius) == 2
            radius = random.uniform(self.radius[0], self.radius[1])
        else:
            raise ValueError
        if self.shape == 'cycle':
            mask = self._hard_cycle_mask(results, radius=radius)
        elif self.shape == 'square':
            mask = self._square_mask(results, radius=radius)
        else:
            raise NotImplementedError

        if self.pass_type == 'high':
            mask = ~mask
        elif self.pass_type == 'random' and self._random_prob() < self.prob:
            mask = ~mask

        img = results['img']
        result_img = np.empty_like(img)
        for i in range(3):
            channel_img = img[:, :, i]
            # Fourier transform
            f = np.fft.fft2(channel_img)
            # Shift the spectrum to the central location
            fshift = np.fft.fftshift(f)
            # filter
            filter_fshift = mask * fshift
            # Shift the spectrum to its original location
            ishift = np.fft.ifftshift(filter_fshift)
            # Inverse Fourier Transform
            iimg = np.fft.ifft2(ishift)
            # kep the input and output have same type
            iimg = np.abs(iimg).astype(np.float32)
            result_img[:, :, i] = iimg
        if self.pass_type == 'none':
            results['fft_filter_img'] = img
        else:
            results['fft_filter_img'] = result_img
        return results
