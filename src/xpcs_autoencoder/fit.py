# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from tqdm import tqdm

from IPython.display import clear_output
from contextlib import contextmanager

from .match_opt import MatchOpt

__all__ = ['run_fit', 'SolverLogbook']


def run_fit(match_opt: MatchOpt, maxiter: int = 1000):
    """
    Run differential evolution routine.

    :param match_opt: MatchOpt object for calculating the target function for optimization.
    :param maxiter: max number of iterations
    :return: tuple of 2 objects:
        result of the DE solver
        SolverLogbook with the history data of the fit
    """
    logbook = SolverLogbook()

    with DifferentialEvolutionSolver(match_opt.get_loss, match_opt.get_bounds(6), maxiter=maxiter) as solver:
        solver.callback = get_de_callback(solver, logbook)
        result = solver.solve()

    return result, logbook


@contextmanager
def get_callback(*args, **kwargs):
    with tqdm(*args, **kwargs) as pbar:
        def func(args, convergence):
            pbar.update(1)
            pbar.set_description(f'convergence {convergence:.3e}')

        yield func


def get_de_callback(solver, logbook):
    iteration = 0

    def func(x, convergence):
        nonlocal iteration

        clear_output(wait=True)
        pop = solver._scale_parameters(solver.population)
        energies = solver.population_energies

        logbook.append(pop, energies)
        logbook.plot_params()
        print(f'Iteration {iteration}')
        iteration += 1

    return func


class SolverLogbook(object):
    def __init__(self, k: int = 20):
        self.k = k
        self.params = []
        self.energies = []

    def __len__(self):
        return len(self.energies)

    @property
    def dsets(self):
        try:
            return (self.params[0].shape[1] - 4) // 2
        except IndexError:
            return 0

    def append(self, population, energies):
        finite_idx = np.isfinite(energies)
        population, energies = population[finite_idx], energies[finite_idx]

        indices = _get_lowest_k(energies, self.k)

        pop, energies = population[indices], energies[indices]
        params = self.restore_params(pop)
        self.params.append(params)
        self.energies.append(energies)

    @staticmethod
    def restore_params(pop):
        tlower, tupper, r, dr = pop[:, -4], pop[:, -3], pop[:, -2], pop[:, -1]
        dset = (pop.shape[1] - 4) // 2
        psigels = pop[:, :dset].swapaxes(0, 1)
        eps0 = pop[:, dset]
        epsilons = [eps0]
        for i in range(dset - 1):
            epsilons.append(epsilons[-1] + pop[:, dset + 1 + i])

        params = np.stack([*epsilons, *psigels, tlower, tupper, r, dr], 1)
        return params

    def epsilons(self, idx: int):
        return self.params[idx][:, :self.dsets]

    def psi_gels(self, idx: int):
        return self.params[idx][:, self.dsets:2 * self.dsets]

    def tlower(self, idx: int):
        return self.params[idx][:, -4]

    def tupper(self, idx: int):
        return self.params[idx][:, -3]

    def r(self, idx: int):
        return self.params[idx][:, -2]

    def dr(self, idx: int):
        return self.params[idx][:, -1]


def plot_param_err(
        epsilons, psi_gels,  # k, dsets
        energies,  # k
        temps=(4, 6, 8, 10, 12, 15),
        cmap: str = 'seismic'
):
    dsets = len(temps)
    cmap = mpl.cm.get_cmap(cmap, dsets)

    eps_means, eps_errs = _get_mean_err(epsilons, energies)
    psi_means, psi_errs = _get_mean_err(psi_gels, energies)

    plt.grid()

    for i, (em, eer, pm, per) in enumerate(zip(eps_means, eps_errs, psi_means, psi_errs)):
        plt.errorbar(x=em, y=pm, xerr=eer, yerr=per, ms=20, c=cmap(i), capsize=20)

    plt.plot(eps_means, psi_means, '--', c='black')
    plt.xlabel('$\epsilon$')
    plt.ylabel('$\psi_{gel}$')


def _get_mean_err(params, energies):
    means = np.einsum('kd,k', params, energies) / energies.sum()
    errs = params.std(0)
    return means, errs


def _get_lowest_k(arr: np.ndarray, k: int):
    return np.argpartition(arr, k)[:k]


def _get_bounds(temps):
    t = np.array(temps)

    bounds = [temps[0] - (temps[1] - temps[0]) / 2]
    bounds += list((t[1:] - t[:-1]) / 2 + t[:-1])
    bounds += [2 * t[-1] - bounds[-1]]
    return bounds
