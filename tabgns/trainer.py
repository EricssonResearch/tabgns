from tabgns.tabgns import TabGNS
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch

class TabGNSTrainer:
  def __init__(self, config: dict):
    self.config = config

  def fit(self, X: np.ndarray, y: np.ndarray, valid_X: np.ndarray, valid_y: np.ndarray):
    self.config['general']['inp_dim'] = X.shape[1]
    
    if self.config['general']['problem_type'] == 'classification':
      self.config['general']['criterion'] = 'cross_entropy'
      if len(y.shape) > 1 and y.shape[1] > 1:
        self.on_hot_encoder = None
        self.config['general']['out_dim'] = y.shape[1]
      else:
        y = y.reshape(-1, 1) if len(y.shape) == 1 else y
        valid_y = valid_y.reshape(-1, 1) if len(valid_y.shape) == 1 else valid_y
        self.on_hot_encoder = OneHotEncoder()
        y = self.on_hot_encoder.fit_transform(y).toarray()
        valid_y = self.on_hot_encoder.transform(valid_y).toarray()
        self.config['general']['out_dim'] = y.shape[1]
    else:
      self.config['general']['criterion'] = 'mse'
      self.config['general']['out_dim'] = y.shape[1] if len(y.shape) > 1 else 1

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    valid_X_tensor = torch.tensor(valid_X, dtype=torch.float32)
    valid_y_tensor = torch.tensor(valid_y, dtype=torch.float32)
    
    train_data = TensorDataset(X_tensor, y_tensor)
    valid_data = TensorDataset(valid_X_tensor, valid_y_tensor)
    
    train_loader = DataLoader(train_data, batch_size=self.config['general']['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=self.config['general']['batch_size'], shuffle=True)
    self.tabgns = TabGNS(self.config)
    self.tabgns.fit(train_loader, valid_loader)

    self.model = self.tabgns.extract_net()

    self.model = self.model.fit(X, y, valid_X, valid_y)

    return self.model

  def predict(self, X: np.ndarray):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    pred = self.model.predict(X_tensor).cpu().numpy()
    if self.config['general']['problem_type'] == 'classification':
      if self.on_hot_encoder is not None:
        return self.on_hot_encoder.inverse_transform(pred).reshape(-1)
      else:
        return pred
    else:
      return pred