import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from typing import Union, List

from tabgns.utils import train_mlp, EarlyStopper


class MLP(nn.Module):
  """ Multi-layer perceptron """
  def __init__(self, config: dict):
    super().__init__()
    self.config = config
    self.inp_dim = config['general']['inp_dim']
    self.out_dim = config['general']['out_dim']
    self.layer_widths = config['layer_widths']
    
    net = []
    for i, width in enumerate(self.layer_widths):
      if i == 0:
        net.append(nn.Linear(self.inp_dim, width))
      else:
        net.append(nn.Linear(self.layer_widths[i-1], width))
      if 'dropout' in config['general'] and config['general']['dropout'] > 0:
        net.append(nn.Dropout(config['general']['dropout']))
      net.append(nn.ReLU())
    
    net.append(nn.Linear(self.layer_widths[-1], self.out_dim))
    self.net = nn.Sequential(*net)
    
  def forward(self, x: Tensor) -> Tensor:
    return self.net(x)
    
  def save(self, path: str):
    torch.save(self.state_dict(), path)
  
  def load(self, path: str):
    self.load_state_dict(torch.load(path))
    
  def fit(self, X: np.ndarray, y: np.ndarray, valid_X: np.ndarray, valid_y: np.ndarray) -> 'MLP':
    if self.config['general']['stop_patience'] == -1:
      earlystopper = None
    else:
      model_copy = MLP(self.config)
      earlystopper = EarlyStopper(patience=self.config['general']['stop_patience'], model=model_copy)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    valid_X_tensor = torch.tensor(valid_X, dtype=torch.float32)
    if self.config['general']['problem_type'] == 'classification':
      y_tensor = torch.tensor(y)
      valid_y_tensor = torch.tensor(valid_y)
      if y_tensor.ndim > 1 and y_tensor.size(-1) > 1:
        y_tensor = torch.argmax(y_tensor, dim=1)
      elif y_tensor.ndim > 1 and y_tensor.size(-1) == 1:
        y_tensor = y_tensor.squeeze(-1)
      if valid_y_tensor.ndim > 1 and valid_y_tensor.size(-1) > 1:
        valid_y_tensor = torch.argmax(valid_y_tensor, dim=1)
      elif valid_y_tensor.ndim > 1 and valid_y_tensor.size(-1) == 1:
        valid_y_tensor = valid_y_tensor.squeeze(-1)
      y_tensor = y_tensor.long()
      valid_y_tensor = valid_y_tensor.long()
    else:
      y_tensor = torch.tensor(y, dtype=torch.float32)
      valid_y_tensor = torch.tensor(valid_y, dtype=torch.float32)
    
    train_data = TensorDataset(X_tensor, y_tensor)
    valid_data = TensorDataset(valid_X_tensor, valid_y_tensor)
    
    train_loader = DataLoader(train_data, batch_size=self.config['general']['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=self.config['general']['batch_size'], shuffle=True)

    optimiser = AdamW(self.parameters(), lr=self.config['general']['weight_lr'])
    criterion = nn.MSELoss() if self.config['general']['problem_type'] == 'regression' else nn.CrossEntropyLoss()

    trained_model = train_mlp(
      model=self,
      criterion=criterion,
      optimiser=optimiser,
      epochs=self.config['general']['iters'],
      train_dl=train_loader,
      valid_dl=valid_loader,
      early_stopper=earlystopper,
      device=self.config['general']['device']
    )
    self.load_state_dict(trained_model.state_dict())
    return self
  
  def predict(self, x: Union[DataLoader, torch.Tensor]) -> torch.Tensor:
    self.to(self.config['general']['device'])
    if isinstance(x, DataLoader):
      predictions = []
      self.eval()
      with torch.no_grad():
        for batch in x:
          batch = batch.to(self.config['general']['device'])
          outputs = self(batch).cpu()
          predictions.append(outputs)
      return torch.cat(predictions, dim=0)
      
    elif isinstance(x, torch.Tensor):
      self.eval()
      with torch.no_grad():
        x = x.to(self.config['general']['device'])
        return torch.argmax(self(x), dim=1)
    else:
      raise TypeError("Input must be either a DataLoader or a torch.Tensor")
      
  def get_architecture(self) -> List[int]:
    return self.layer_widths