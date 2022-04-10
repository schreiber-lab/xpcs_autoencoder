# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

from .exp_dataset import EncodedExpDataset
from .sim_dataset import SimDataset

__all__ = ['MatchOpt']


class MatchOpt(object):
    """
    Calculates the match function - a mean dot product ξ between the normalized encoded representations of 30
    experimental TTCs and 30 simulated TTCs from the training dataset.
    """

    def __init__(self, exp_dset: EncodedExpDataset, sim_dset: SimDataset):
        self.exp_dset = exp_dset
        self.sim_dset = sim_dset

    def get_match(self, epsilons, psi_gels, t_lower, t_upper, r, dr):
        rs = [r - 2 * dr, r - dr, r, r + dr, r + 2 * dr]
        if rs[0] < 0 or rs[-1] > 19:
            return - np.inf
        try:
            zs_sim = torch.stack([self.sim_dset.get_z_lin((eps, psi), rs) for eps, psi in zip(epsilons, psi_gels)], 0)
            zs_exp = self.exp_dset.get_joint_z(int(t_lower), int(t_upper))
        except (ValueError, IndexError):
            return - np.inf
        return torch.einsum('ijk,ijk', zs_exp, zs_sim).item()

    def get_loss(self, args):
        epsilons, psi_gels, t_lower, t_upper, r, dr = self.parse_args(args)

        if epsilons[-1] > 0.99:
            return np.inf

        return - self.get_match(epsilons, psi_gels, t_lower, t_upper, r, dr)

    @staticmethod
    def parse_args(args):
        t_lower, t_upper, r, dr = args[-4:]
        params = args[:-4]
        dset_num = len(params) // 2
        psi_gels = params[:dset_num]
        eps0 = params[dset_num]
        eps_delta = params[dset_num + 1:]

        epsilons = [eps0]

        for d in eps_delta:
            epsilons.append(epsilons[-1] + d)

        return epsilons, psi_gels, t_lower, t_upper, r, dr

    @staticmethod
    def get_bounds(dset_num: int):
        psi_gel_b = (0.55, 1)
        eps0_b = (0.88, 0.95)
        d_eps_b = (0.0005, 0.05)
        t_lower_b = (0, 200)
        t_upper_b = (1200, 3000)
        r_b = (3, 16)
        dr_b = (0.1, 2)

        bounds = (
                [psi_gel_b] * dset_num +
                [eps0_b] +
                [d_eps_b] * (dset_num - 1) +
                [t_lower_b, t_upper_b, r_b, dr_b]
        )

        return tuple(bounds)
