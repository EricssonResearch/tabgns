"""
Tools to train and evaluate models: Training functions, metrics for evaluation, model unrolling, etc.
"""
import os
import random
from tqdm.auto import tqdm
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.nn import Module

from typing import Optional, List


def accuracy(y_true: Tensor, y_pred: Tensor) -> float:
  y_true = torch.argmax(y_true, dim=1)
  y_pred = torch.argmax(y_pred, dim=1)
  return (y_true == y_pred).float().mean().item()

def param_count(arch: List[int]) -> int:
  return sum([arch[i] * arch[i+1] for i in range(len(arch)-1)])
  
class EarlyStopper:
  """Early stopper that also saves the best model and then can load it back."""
  def __init__(self, patience: int, model: Module):
    print(f'Early stopper initialized. Patience: {patience}')
    self.patience = patience
    self.counter = 0
    self.best = np.inf
    self.stop = False
    self.best_model: Module = model
    
  def __call__(self, val_loss: float, model: Module):
    if val_loss < self.best:
      self.best = val_loss
      self.counter = 0
      self.best_model.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
    else:
      self.counter += 1
    if self.counter >= self.patience:
      self.stop = True
      print('Early stopping')
    return self.stop
    
  def load_best(self) -> Module:
    return self.best_model

def train_mlp(model: Module,
    criterion: Module,
    optimiser: Optimizer,
    epochs: int,
    train_dl: DataLoader,
    valid_dl: Optional[DataLoader]=None,
    early_stopper: Optional[EarlyStopper] = None,
    device: str = 'cpu'
) -> Module:
  if isinstance(criterion, str):
    if criterion.lower() in {"mse", "mse_loss"}:
      criterion = nn.MSELoss()
    elif criterion.lower() in {"cross_entropy", "ce", "crossentropyloss"}:
      criterion = nn.CrossEntropyLoss()
    else:
      raise ValueError(f"Unknown criterion string: {criterion}")
  pbar = tqdm(range(epochs), desc="Training")
  model = model.to(device)
  for e in pbar:
    model.train()
    train_loss = 0.
    for X, y in train_dl:
      X, y = X.to(device), y.to(device)
      if isinstance(criterion, nn.CrossEntropyLoss):
        if y.ndim > 1 and y.size(-1) > 1:
          y = torch.argmax(y, dim=1)
        elif y.ndim > 1 and y.size(-1) == 1:
          y = y.squeeze(-1)
        y = y.long()
      optimiser.zero_grad()
      out = model(X)
      loss = criterion(out, y)
      loss.backward()
      optimiser.step()
      train_loss += loss.item()
    train_loss /= len(train_dl)

    valid_loss = 0.
    if valid_dl is not None:
      model.eval()
      with torch.no_grad():
        for X, y in valid_dl:
          X, y = X.to(device), y.to(device)
          if isinstance(criterion, nn.CrossEntropyLoss):
            if y.ndim > 1 and y.size(-1) > 1:
              y = torch.argmax(y, dim=1)
            elif y.ndim > 1 and y.size(-1) == 1:
              y = y.squeeze(-1)
            y = y.long()
          out = model(X)
          loss = criterion(out, y)
          valid_loss += loss.item()
      valid_loss /= len(valid_dl)

      if early_stopper is not None:
        if early_stopper(valid_loss, model):
          break

    pbar_desc = f'Epoch {e+1}/{epochs}, Train Loss: {train_loss:.4f}'
    if valid_dl is not None:
      pbar_desc += f', Valid Loss: {valid_loss:.4f}'
    pbar.set_description(pbar_desc)
  model = early_stopper.load_best() if early_stopper is not None else model
  return model
  
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
