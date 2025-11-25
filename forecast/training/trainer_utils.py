import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple


def create_dataloader(X, y, batch_size: int, shuffle: bool = True):
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float32)

    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def move_batch_to_device(batch, device):
    X, y = batch
    return X.to(device), y.to(device)


def compute_loss(pred, target, loss_fn):
    return loss_fn(pred, target)
