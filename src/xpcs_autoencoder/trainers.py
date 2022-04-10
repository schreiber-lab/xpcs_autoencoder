# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
from pathlib import Path

import torch
from torch.optim import AdamW
from torch import nn
from torch.optim import lr_scheduler

import numpy as np

from tqdm import tqdm

from .tools import Losses, clear_output
from .loaders import DataLoader, AEDataLoader
from .autoencoder import AutoEncoder

__all__ = ['train_autoencoder']


def train_autoencoder(autoencoder: AutoEncoder, dataloader: AEDataLoader, name: str = 'autoencoder.pt',
                      trainer: 'AETrainer' = None, num_epochs: int = 30, save_every_epoch: bool = True):
    trainer = trainer or AETrainer(autoencoder, dataloader, 2e-4)
    ls_scheduler = ExpDecayLRCallback(0.99)
    trainer.train(num_epochs, (
        SaveBestModel(name, save_every_epoch=save_every_epoch), ls_scheduler
    ))
    trainer.plot()
    return trainer


class Trainer(object):
    def __init__(self, model, loader: DataLoader, lr: float):
        self.model = model
        self.loader = loader
        self.optim = AdamW(model.parameters(), lr=lr)

        self.losses = Losses()

        self.callback_params = {}

        self._init_criteria()

    def _init_criteria(self):

        self.criterion = nn.MSELoss()

    def train_epoch(self, epoch: int, num_epochs: int = 1):

        self.model.train()

        for i, batch_data in enumerate(tqdm(
                self.loader.train(),
                total=self.loader.train_size // self.loader.batch_size,
                desc=f'Epoch {epoch + 1} / {num_epochs}')
        ):
            self.optim.zero_grad()

            loss, loss_dict = self._get_loss(batch_data)
            loss.backward()
            self.optim.step()

            if i % 10 == 0:
                clear_output(wait=True)
                self.losses.plot()

            self._add_batch_losses({'train': loss.item(), **{f'train_{k}': v for k, v in loss_dict.items()}})

    def _get_loss(self, batch_data):
        ttcs, cuts, labels = batch_data
        loss = self.criterion(self.model(ttcs, cuts), labels)
        return loss, {}

    def val(self):

        self.model.eval()

        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(
                    self.loader.val(),
                    total=self.loader.val_size // self.loader.batch_size,
                    desc=f'Val epoch')
            ):
                loss, loss_dict = self._get_loss(batch_data)
                self._add_batch_losses({'val': loss.item(), **{f'val_{k}': v for k, v in loss_dict.items()}})

    def _add_batch_losses(self, loss_dict):
        for k, v in loss_dict.items():
            self.losses[f'batch_{k}'].append(v)

    def _reduce_batch_losses(self):
        to_delete = []
        to_append = []

        for k, v in self.losses.items():
            if k.startswith('batch_'):
                to_append.append((k[6:], np.mean(self.losses[k])))
                to_delete.append(k)

        for k in to_delete:
            del self.losses[k]

        for k, v in to_append:
            self.losses[k].append(v)

    def plot(self):
        clear_output(wait=True)
        self.losses.plot(self.callback_params.get('best_epoch', None))

    def train(self, num_epochs: int, callbacks: Tuple['TrainerCallback', ...] = ()):

        for c in callbacks:
            c.start_training(self)

        for epoch in range(num_epochs):

            self.train_epoch(epoch, num_epochs)
            self.val()
            self._reduce_batch_losses()

            stop_training = any([c.end_epoch(self, epoch) for c in callbacks])

            self.plot()

            if stop_training:
                break

        for c in callbacks:
            c.end_training(self)

    def set_lr(self, lr: float):
        self.optim.param_groups[0]['lr'] = lr

    def get_lr(self):
        return self.optim.param_groups[0]['lr']


class AETrainer(Trainer):
    def _get_loss(self, ttc):
        predicted_ttc = self.model(ttc)
        loss = self.criterion(torch.tril(predicted_ttc), torch.tril(ttc))
        return loss, {}


class TrainerCallback(object):
    def start_training(self, trainer: Trainer) -> None:
        pass

    def end_training(self, trainer: Trainer) -> None:
        pass

    def end_epoch(self, trainer: Trainer, epoch: int) -> bool:
        return False


class SaveBestModel(TrainerCallback):
    def __init__(self, model_path: Path or str = None,
                 load_best_model: bool = False,
                 init_loss: float = None,
                 save_every_epoch: bool = False,
                 loss_key: str = 'val'
                 ):
        self._state_dict = None
        self._loss_key = loss_key
        self._val_loss = init_loss
        self._best_epoch = 0
        self._load_best_model = load_best_model
        self.model_path = Path(model_path)
        self.save_every_epoch = save_every_epoch

    def clear(self):
        self._state_dict = None
        self._val_loss = None
        self._best_epoch = 0

    def save_model(self):
        if self.model_path and self._state_dict:
            torch.save(self._state_dict, self.model_path)

    def load_best_model(self, trainer):
        if self._state_dict:
            trainer.model.load_state_dict(self._state_dict)

    def start_training(self, trainer: Trainer):
        self.clear()
        if 'best_epoch' in trainer.callback_params:
            del trainer.callback_params['best_epoch']

    def end_training(self, trainer: Trainer):
        if self._load_best_model:
            self.load_best_model(trainer)

        if not self.save_every_epoch:
            self.save_model()

    def end_epoch(self, trainer: Trainer, epoch: int):
        if self._val_loss is None or self._val_loss > trainer.losses[self._loss_key][-1]:
            self._val_loss = trainer.losses[self._loss_key][-1]
            self._state_dict = trainer.model.cpu_state_dict()
            self._best_epoch = epoch
            trainer.callback_params['best_epoch'] = self._best_epoch

            if self.save_every_epoch:
                self.save_model()

        return False


class LRSchedulerCallback(TrainerCallback):
    def __init__(self, scheduler_class, **kwargs):
        self._scheduler_class = scheduler_class
        self._kwargs = kwargs
        self.lr_list = []

    def start_training(self, trainer: Trainer) -> None:
        self.scheduler = self._scheduler_class(trainer.optim, **self._kwargs)

    def end_epoch(self, trainer: Trainer, epoch: int) -> bool:
        self.scheduler.step()
        self.lr_list.append(trainer.get_lr())

        return False

    def clear(self):
        self.lr_list.clear()


class ExpDecayLRCallback(LRSchedulerCallback):
    def __init__(self, gamma: float = 0.99, **kwargs):
        kwargs['gamma'] = gamma
        super().__init__(lr_scheduler.ExponentialLR, **kwargs)


class VectorsTrainer(Trainer):
    def _get_loss(self, batch_data):
        vectors, labels = batch_data
        loss = self.criterion(self.model(vectors), labels)
        return loss, {}
