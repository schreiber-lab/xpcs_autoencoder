# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from .package_info import *
from .autoencoder import init_autoencoder
from .sim_dataset import get_sim_dataset
from .exp_dataset import get_exp_ttc_dset
from .match_opt import MatchOpt
from .fit import run_fit
from .trainers import train_autoencoder
from .loaders import AEDataLoader


def run(sim_path: str, exp_path: str):
    """
    Trains the autoencoder, initializes the datasets and performs the fit.

    :param sim_path: path to a h5 file with the simulated TTCs
    :param exp_path: path to a h5 file with the experimental TTCs
    :return: tuple with 2 objects:
        - TTC autoencoder
        - results of the fit
    """

    ####### Train the autoencoder ########

    # Initialize the autoencoder.
    auto_encoder = init_autoencoder()

    # Initialize the data loader of the simulated TTCs (two-time correlation maps) from a h5 file.
    # The corresponding CH simulations can be generated via https://github.com/StarostinV/cahnhilliard package,
    # and the TTCs can be calculated via ttc_calculation.py module of this package.
    sim_dataloader = AEDataLoader(sim_path)

    # Train the autoencoder on the simulated TTCs and save to autoencoder.pt file.
    train_autoencoder(auto_encoder, sim_dataloader, name='autoencoder.pt')

    ####### Perform the fit of the experimental TTCs ########

    # Initialize the simulated and the experimental datasets with TTCs.
    sim_dset = get_sim_dataset(sim_path, auto_encoder)
    exp_dset = get_exp_ttc_dset(exp_path, auto_encoder)

    # Initialize the MatchOpt object to calculate the measure of a difference between TTCs in the encoded space.
    match_optim = MatchOpt(exp_dset, sim_dset)

    # Perform the fit.
    results, fit_history = run_fit(match_optim)

    return auto_encoder, results
