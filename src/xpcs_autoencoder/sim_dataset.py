# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from scipy.spatial import KDTree
from h5py import File
from tqdm import tqdm

from .autoencoder import AutoEncoder

__all__ = ['get_sim_dataset', 'get_encoded_ttcs', 'SimDataset']


def get_sim_dataset(data_path: str, autoencoder: AutoEncoder) -> 'SimDataset':
    """
    Get dataset with the encoded simulated TTCs.

    :param data_path: path to an h5 file with the simulated TTCs
    :param autoencoder: autoencoder
    :return: SimDataset
    """
    return SimDataset.from_path(data_path, autoencoder.encoder)


def get_encoded_ttcs(data_path: str, encoder):
    """
    Returns the encoded representation of the simulated TTCs from an h5 file along with the simulation parameters,
    cut-off positions and the group keys of the h5 file. CH simulations can be generated via the following package:
    https://github.com/StarostinV/cahnhilliard

    :param data_path: path to the h5 file with the simulated TTCs.
    :param encoder: TTC encoder
    :return: tuple of 4 arrays:
        zs_dataset - encoded representation of the simulated TTCs from the file
        params_dataset - the corresponding parameters of the simulation
        peak_indices - time indices corresponding to the cut-off positions
        keys -
    """
    zs = []
    params = []
    peak_indices = []
    keys = []

    encoder.eval()

    with File(data_path, 'r') as f:
        with torch.no_grad():
            for k, g in tqdm(f.items()):
                keys.append(k)
                zs.append(encoder(torch.tensor(g['ttcs'][()][:, None],
                                               dtype=torch.float32, device='cuda')
                                  ).squeeze().cpu().numpy())

                params.append((g['eps'][()], g['u_gel'][()]))
                peak_indices.append(g['peak_indices'][()])

    zs_dataset, params_dataset, peak_indices = np.array(zs), np.array(params), np.array(peak_indices)

    return zs_dataset, params_dataset, peak_indices, keys


class SimDataset(object):
    """
    Dataset with simulated TTCs.
    """

    def __init__(self, params_dataset, zs_dataset, device: str = 'cuda'):
        self.params_dataset = params_dataset
        self.device = device
        self.zs_dataset = torch.tensor(zs_dataset, device=device)  # num, r, z
        self.tree = KDTree(params_dataset)

    @classmethod
    def from_path(cls, sim_path: str, encoder, device: str = 'cuda'):
        zs_dataset, params_dataset, peak_indices, keys = get_encoded_ttcs(sim_path, encoder)
        return cls(params_dataset, zs_dataset, device=device)

    def get_idx(self, params):
        return self.tree.query(params, )[-1]

    def get_z(self, rs):
        rl, ru = np.floor(rs), np.ceil(rs)
        alphas = torch.tensor((rs - rl)[..., None], device=self.device, dtype=torch.float32)

        return self.zs_dataset[:, ru.astype(int)] * alphas + self.zs_dataset[:, rl.astype(int)] * (1 - alphas)

    def get_z_lin(self, params, rs):
        idx = self.get_idx(params)

        rl, ru = np.floor(rs), np.ceil(rs)
        alphas = torch.tensor((rs - rl)[..., None], device=self.device, dtype=torch.float32)

        return self.zs_dataset[idx, ru.astype(int)] * alphas + self.zs_dataset[idx, rl.astype(int)] * (1 - alphas)

    def get_closest_params(self, params):
        loss, idx = self.tree.query(params)
        return self.params_dataset[idx], loss, idx
