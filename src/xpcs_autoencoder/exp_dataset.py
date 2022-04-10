# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
from h5py import File
from tqdm import tqdm
from cv2 import resize

import torch
from torch import Tensor

from .autoencoder import AutoEncoder

__all__ = ['get_exp_ttc_dset', 'EncodedExpDataset']


def get_exp_ttc_dset(exp_path: Path or str, autoencoder: AutoEncoder) -> 'EncodedExpDataset':
    """
    Get EncodedExpDataset object that calculates the encoded experimental TTCs.

    :param exp_path: path to an h5 file with the saved experimental TTCs
    :param autoencoder: autoencoder model for TTCs
    :return: EncodedExpDataset object
    """
    exp_dset = ExperimentalDataset(exp_path)
    exp_ttc_dset = EncodedExpDataset(autoencoder, exp_dset)
    return exp_ttc_dset


class ExperimentalDataset(object):

    def __init__(self, path: Path or str,
                 raw_ttc_key: str = 'ttc',
                 processed_ttc_key: str = 'processed_ttc',
                 ):
        """
        API to the dataset with the experimental TTCs saved to an h5 file.
        :param path: a path to an h5 file with the experimental TTCs
        :param raw_ttc_key: h5 key to the raw TTCs
        :param processed_ttc_key: h5 key to the processed TTCs
        """

        self.path = Path(path)
        self.raw_ttc_key = raw_ttc_key
        self.processed_ttc_key = processed_ttc_key

        with File(self.path, 'r') as f:
            self.datasets = list(f.keys())
            self.dataset_keys = list(f[self.datasets[0]].keys())

            self.rs = []

            for k in self.dataset_keys:
                try:
                    r = int(k)
                except ValueError:
                    continue

                self.rs.append(r)

            self.ttc_keys = list(f[self.datasets[0]][str(self.rs[0])].keys())

        self._file = None

        assert self.raw_ttc_key in self.ttc_keys
        assert self.processed_ttc_key in self.ttc_keys

    def get_data(self, ttc_key: str, r: int, dset: int, ttc_slice: tuple or slice = ()):

        r = str(self.rs[r])
        dset = str(self.datasets[dset])

        with File(self.path, 'r') as f:
            return f[dset][r][ttc_key][ttc_slice]

    def get_raw_ttc(self, r: int, dset: int, ttc_slice: slice = ...):
        return self.get_data(self.raw_ttc_key, r, dset, ttc_slice)

    def get_processed_ttc(self, r: int, dset: int, ttc_slice: slice = ...):
        return self.get_data(self.processed_ttc_key, r, dset, ttc_slice)

    @contextmanager
    def __call__(self, dset: int = None, r: int = None):
        with File(self.path, 'r') as group:

            if dset is not None:
                group = group[str(self.datasets[dset])]

                if r is not None:
                    group = group[str(self.rs[r])]

            yield group


class EncodedExpDataset(object):
    def __init__(self, autoencoder, exp_dset: ExperimentalDataset, *,
                 rs: tuple = (1, 3, 5, 7, 9), dsets: tuple = (0, 1, 2, 3, 4, 5)):

        self.autoencoder = autoencoder
        self.exp_dset = exp_dset
        self.rs = np.array(rs)
        self.dsets = np.array(dsets)

        self.ttcs = [
            [
                self.exp_dset.get_processed_ttc(r, d)
                for r in rs
            ] for d in tqdm(dsets)
        ]

        self.peak_idx = []

        for d in dsets:

            peaks = []

            for r in rs:
                with self.exp_dset(d, r) as f:
                    idx = f['idx'][()]
                    peaks.append(idx)

            self.peak_idx.append(peaks)

    @lru_cache(maxsize=1024)
    def get_joint_z(self, t_lower: int, t_upper: int):

        ttcs = self.get_ttcs(t_lower, t_upper)

        zs = []

        with torch.no_grad():
            for ttc_group in ttcs:
                ttc_tensor = torch.tensor(ttc_group, dtype=torch.float32, device='cuda')[:, None]
                z = self.autoencoder.encoder(self.autoencoder(ttc_tensor))

                z /= torch.sqrt((z ** 2).sum(-1))[..., None]

                zs.append(z)

        return torch.stack(zs)

    def get_ttcs(self, t_lower: int, t_upper: int):
        return [
            [
                resize(
                    _renormalize_ttc(self.ttcs[i][j][
                                     t_lower:t_upper - self.peak_idx[i][j],
                                     t_lower:t_upper - self.peak_idx[i][j]]
                                     ),
                    (64, 64)
                )
                for j in range(len(self.rs))
            ] for i in range(len(self.dsets))
        ]


def _set_diag(arr, value: float = 1.):
    for i in range(arr.shape[-1]):
        arr[i, i] = value


def _renormalize_ttc(ttc: Tensor or np.ndarray):
    _set_diag(ttc, ttc.min())

    ttc /= ttc.max()

    _set_diag(ttc)

    return ttc
