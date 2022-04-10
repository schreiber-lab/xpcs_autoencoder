# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch


def calculate_2d_corr(intensity_array: np.ndarray) -> np.ndarray:
    """
    Calculate a two-time correlation map.

    intensity_array.shape == (time_axis, pixel_axis)
    """
    std_array = intensity_array.std(axis=1)[:, np.newaxis]
    std_array[std_array == 0] = 1
    intensity_array /= std_array
    res = intensity_array.dot(intensity_array.transpose()) / intensity_array.shape[1]
    i_mean = np.expand_dims(intensity_array.mean(axis=1), axis=1)
    res -= i_mean.dot(i_mean.transpose())
    return res


def calculate_2d_corr_torch(intensity_array: torch.Tensor) -> torch.Tensor:
    """
    Calculate a batch of two-time correlation maps.

    intensity_array.shape == (batch_axis, time_axis, pixel_axis)
    """
    std_array = torch.std(intensity_array, -1, unbiased=False)[..., None]
    std_array[std_array == 0] = 1
    intensity_array /= std_array
    res = torch.bmm(intensity_array, intensity_array.transpose(1, 2)) / intensity_array.shape[-1]
    i_mean = intensity_array.mean(axis=-1)[..., None]
    res -= torch.bmm(i_mean, i_mean.transpose(1, 2))
    return res


class TTCCalculator(object):
    def __init__(self, qs: tuple = tuple(range(26, 46)), nx: int = 256, dq: float = 1.3):
        self.qs = qs
        self.dq = dq
        self.size = nx
        self._rr = get_rr(nx, use_torch=False)
        self._init_masks()

    def _init_coords(self):
        self._x = np.arange(self.size) - self.size / 2
        self._y = np.array(self._x)
        self._xx, self._yy = np.meshgrid(self._x, self._y)
        self._rr = np.sqrt(self._xx ** 2 + self._yy ** 2)

    def _init_masks(self):
        self.masks = {q: get_mask(self._rr, q, self.dq) for q in self.qs}

    def get_intensity(self, ft: np.ndarray, q: int):
        return ft[:, self.masks[q]]

    def get_ttc(self, ft, q: int):
        return calculate_2d_corr(self.get_intensity(ft, q))


def get_rr(size: int, use_torch: bool = True):
    if use_torch:
        module = torch
    else:
        module = np

    x = module.arange(size, dtype=module.float32) - size / 2
    xx, yy = module.meshgrid(x, x)
    rr = module.sqrt(xx ** 2 + yy ** 2)

    if use_torch:
        rr = rr.cuda()

    return rr


def get_mask(rr: np.ndarray or torch.Tensor, r: float, dr: float):
    return (rr >= r - dr) & (rr <= r + dr)
