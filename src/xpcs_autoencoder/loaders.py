# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
from pathlib import Path
from itertools import product

import numpy as np
import torch

from h5py import File

__all__ = ['AEDataLoader']


class LabelsScaler(object):
    def __init__(self, *ranges: Tuple[float, float]):
        self.min_vector = np.array([r[0] for r in ranges])
        self.delta_vector = np.array([r[1] - r[0] for r in ranges])
        self.delta_vector[self.delta_vector == 0] = 1

    def normalize(self, x):
        return (x - self.min_vector) / self.delta_vector

    def rescale(self, x):
        return x * self.delta_vector + self.min_vector


class BasicScaler(LabelsScaler):
    def __init__(self,
                 eps_range: Tuple[float, float] = (0.88, 0.99),
                 phi_range: Tuple[float, float] = (0.55, 1),
                 q_range: Tuple[float, float] = (26, 46)
                 ):
        super().__init__(eps_range, phi_range, q_range)


class CutDataScaler(LabelsScaler):
    def __init__(self, cut_range: Tuple[float, float] = (120, 290)):
        super().__init__(cut_range)


class TTCStdScaler(object):
    def __init__(self, mean_ttc: np.ndarray, std_ttc: np.ndarray):
        self.mean_ttc = mean_ttc[None]
        self.std_ttc = std_ttc[None]

    def scale(self, ttcs: np.ndarray):
        return (ttcs - self.mean_ttc) / self.std_ttc

    def restore(self, ttcs: np.ndarray):
        return ttcs * self.std_ttc + self.mean_ttc


class DataLoader(object):
    def __init__(self,
                 file_path: Path or str, batch_size: int = 64,
                 labels_scaler: LabelsScaler = None,
                 cut_scaler: CutDataScaler = None,
                 ttc_scaler: TTCStdScaler = None,
                 train_share: float = 0.7,
                 val_share: float = 0.2,
                 q_min: float = 26
                 ):

        self.labels_scaler = labels_scaler or BasicScaler()
        self.cut_scaler = cut_scaler or CutDataScaler()
        self.ttc_scaler = ttc_scaler

        self.q_min = q_min

        self.batch_size = batch_size
        self.file_path = Path(file_path)

        with File(file_path, 'r') as f:
            self.keys = list(map(int, f.keys()))
            self.data_shape = f['0/ttcs'].shape

        self._qs = tuple(range(self.data_shape[0]))

        self.size = self._get_batch_num(len(self.keys) * len(self._qs))

        self._init_keys(train_share, val_share)

    def _init_keys(self, train_share, val_share):
        assert train_share + val_share < 1.

        self.train_size = int(len(self.keys) * train_share)
        self.val_size = int(len(self.keys) * val_share)
        self.test_size = len(self.keys) - self.train_size - self.val_size

        self.train_keys = self.keys[:self.train_size]
        self.val_keys = self.keys[self.train_size:self.train_size + self.val_size]
        self.test_keys = self.keys[self.train_size + self.val_size:]

        self._train_indices = np.array(list(product(self.train_keys, self._qs)))
        self._val_indices = np.array(list(product(self.val_keys, self._qs)))
        self._test_indices = np.array(list(product(self.test_keys, self._qs)))

        self.train_size *= len(self._qs)
        self.val_size *= len(self._qs)
        self.test_size *= len(self._qs)

    def _get_batch(self, indices):
        with File(self.file_path, 'r') as f:
            ttcs = [f[str(idx)]['ttcs'][q] for idx, q in indices]
            labels = np.array([
                (
                    f[str(idx)]['eps'][()],
                    f[str(idx)]['u_gel'][()],
                    q + self.q_min
                )
                for idx, q in indices
            ])

            peak_pos = np.array([f[str(idx)]['peak_indices'][q] for idx, q in indices])

        labels = self.labels_scaler.normalize(labels)
        peak_pos = self.cut_scaler.normalize(peak_pos)[..., None]

        if self.ttc_scaler:
            ttcs = self.ttc_scaler.scale(ttcs)

        return (
            torch.tensor(ttcs, device='cuda', dtype=torch.float32).unsqueeze(1),
            torch.tensor(peak_pos, device='cuda', dtype=torch.float32),
            torch.tensor(labels, device='cuda', dtype=torch.float32),
        )

    def train(self):
        yield from self._iterate(self._train_indices)

    def val(self):
        yield from self._iterate(self._val_indices)

    def test(self):
        yield from self._iterate(self._test_indices)

    def _iterate(self, indices: np.ndarray):
        np.random.shuffle(indices)

        size = self._get_batch_num(indices.shape[0])

        for i in range(size):
            yield self._get_batch(indices[i * self.batch_size:(i + 1) * self.batch_size])

    def __iter__(self):
        yield from self.train()

    def __len__(self):
        return self.size

    def _get_batch_num(self, samples_num: int):
        return samples_num // self.batch_size + (1 if samples_num % self.batch_size != 0 else 0)


class BasicDataLoader(DataLoader):
    def _get_batch(self, indices):
        with File(self.file_path, 'r') as f:
            ttcs = [f[str(idx)]['ttcs'][q] for idx, q in indices]
            labels = np.array([
                (
                    f[str(idx)]['eps'][()],
                    f[str(idx)]['u_gel'][()],
                    q + self.q_min
                )
                for idx, q in indices
            ])

        labels = self.labels_scaler.normalize(labels)

        if self.ttc_scaler:
            ttcs = self.ttc_scaler.scale(ttcs)

        return (
            torch.tensor(ttcs, device='cuda', dtype=torch.float32).unsqueeze(1),
            torch.tensor(labels, device='cuda', dtype=torch.float32),
        )


class VectorsDataLoader(DataLoader):
    def __init__(self,
                 file_path: Path, batch_size: int = 64,
                 labels_scaler: LabelsScaler = None,
                 cut_scaler: CutDataScaler = None,
                 ttc_scaler: TTCStdScaler = None,
                 train_share: float = 0.7,
                 val_share: float = 0.2,
                 q_min: float = 26,
                 vector_key: str = 'resnet_vectors_3'
                 ):
        super().__init__(
            file_path, batch_size,
            labels_scaler, cut_scaler, ttc_scaler,
            train_share, val_share,
            q_min
        )

        self.vector_key = vector_key

    def _get_batch(self, indices):
        with File(self.file_path, 'r') as f:
            vectors = [f[str(idx)][self.vector_key][q] for idx, q in indices]
            labels = np.array([
                (
                    f[str(idx)]['eps'][()],
                    f[str(idx)]['u_gel'][()],
                    q + self.q_min
                )
                for idx, q in indices
            ])

        labels = self.labels_scaler.normalize(labels)

        return (
            torch.tensor(vectors, device='cuda', dtype=torch.float32),
            torch.tensor(labels, device='cuda', dtype=torch.float32),
        )


class AEDataLoader(DataLoader):
    def _get_batch(self, indices):
        with File(self.file_path, 'r') as f:
            ttcs = [f[str(idx)]['ttcs'][q] for idx, q in indices]

        if self.ttc_scaler:
            ttcs = self.ttc_scaler.scale(ttcs)

        return torch.tensor(ttcs, device='cuda', dtype=torch.float32).unsqueeze(1)
