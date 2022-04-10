# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

from matplotlib import pyplot as plt

try:
    from IPython.display import clear_output
except ImportError:
    clear_output = lambda *x: None

__all__ = ['Losses', 'get_size_str']


class Losses(defaultdict):
    def __init__(self):
        super().__init__(list)

    def plot(self, best_epoch: float = None, **kwargs):
        plot_losses(self, best_epoch, **kwargs)


def plot_losses(losses, best_epoch: float = None, log: bool = True):
    func = plt.semilogy if log else plt.plot

    for k, data in losses.items():
        func(data, label=k)

    if best_epoch is not None:
        plt.axvline(best_epoch, ls='--', color='red')

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Losses')
    plt.show()


def get_size_str(size):
    if size > 1000 ** 3:
        size = f'{(size / 1000 ** 3):.2f} Gb'
    elif size > 1000 ** 2:
        size = f'{(size / 1000 ** 2):.2f} Mb'
    else:
        size = f'{(size / 1000):.2f} Kb'
    return size
