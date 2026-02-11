import openml
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

from typing import List, Optional, Tuple

def get_dataset(id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
  dataset = openml.datasets.get_dataset(id,
                                     download_data=True,
                                     download_qualities=True,
                                     download_features_meta_data=True)
  X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format='dataframe')
  return X, y

def load_higgs(small=True) -> Tuple[np.ndarray, np.ndarray]:
  X, y = get_dataset(23512 if small else 44129)
  nan_indices = X[X.isna().any(axis=1)].index
  X.dropna(inplace=True)
  y = y.astype(int)
  # scaler = MinMaxScaler()
  scaler = QuantileTransformer(output_distribution='normal', random_state=0)
  X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
  X, y = X.to_numpy(), y.to_numpy()
  y = np.delete(y, nan_indices)
  return X, y

def load_covertype() -> Tuple[np.ndarray, np.ndarray]:
  X, y = get_dataset(1596)
  nan_indices = X[X.isna().any(axis=1)].index
  X.dropna(inplace=True)
  y = y.astype(int)
  # scaler = MinMaxScaler()
  scaler = QuantileTransformer(output_distribution='normal', random_state=0)
  X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
  X, y = X.to_numpy(), y.to_numpy()
  y = np.delete(y, nan_indices)
  y = one_hot_encode(y)
  return X, y

def split_train_valid_test(x, y, dl=True, batch_size=64, seed=1234):
  print('seed:', seed)
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
  x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=seed)

  if dl:
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader
  else:
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def one_hot_encode(y):
  oh = OneHotEncoder()
  y = y.reshape(-1, 1)
  oh.fit(y)
  encoded = oh.transform(y).toarray()
  return encoded
  