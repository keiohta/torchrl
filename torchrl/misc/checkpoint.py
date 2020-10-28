import collections
import os
from queue import Queue
import time

import torch


def CheckpointManager(object):
    def __init__(self, directory, max_to_keep, ckpt_name='ckpt'):
        self._max_to_keep = max_to_keep
        self._directory = directory
        self.checkpoint_prefix = os.path.join(directory, ckpt_name)

        self._maybe_delete = []

    @property
    def directory(self):
        return self._directory

    @staticmethod
    def _check_exceed_max_size(self):
        return len(self._maybe_delete) == 5

    def _pop_queue(self):
        return self._maybe_delete.pop(0)

    def _pop_specific_queue(self, idx=None, key=None):
        if idx is not None:
            return self._maybe_delete.pop(idx)
        elif key is not None:
            for idx, item in enumerate(self._maybe_delete):
                if item == key:
                    return self._maybe_delete.pop(idx)
        return None

    def _push_to_queue(self, item):
        self._maybe_delete.append(item)

    def save(self, step, **kwargs):
        path = self._checkpoint_prefix + f"_{step}.pt"
        torch.save(kwargs, path)

        if self._check_exceed_max_size:
            to_delete_path = self._pop_queue()
            os.remove(to_delete_path)

        self._maybe_delete.put(path)

    def load(self, model, optim, idx=None, key=None):
        if idx is not None or key is not None:
            path = self._pop_specific_queue(idx=idx, key=key)
        else:
            path = self._pop_queue()

        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model_state_dict'])
        optim.load_state_dict(ckpt['optim_state_dict'])
        epoch = ckpt['epoch']

        return model, optim, epoch
