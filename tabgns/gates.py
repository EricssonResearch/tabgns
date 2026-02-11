"""
A block containing a linear layer, an activation function and a gate using Gumbel softmax.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List


class BinaryGumbelSoftmax(nn.Module):
  """ Gumbel softmax gate """
  def __init__(self, temperature: float=1.0, hard: bool=True):
    super().__init__()
    self.temperature = temperature
    self.hard = hard

  def forward(self, strength: Tensor) -> Tensor:
    logits = torch.stack([strength, torch.ones_like(strength)], dim=1)
    out = F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard)
    return out[:, 0]

class GateFn(nn.Module):
  """ General class for gate functions """
  def __init__(self, temperature: float=1.):
    super().__init__()
    self.gates = BinaryGumbelSoftmax(temperature, hard=True)

  def forward(self, x: Tensor, strengths: Tensor) -> Tensor:
    return x * self.gates(strengths)

class GatedBlock(nn.Module):
  """ Block of linear layer, activation and gate """
  def __init__(self,
               in_features: int,
               out_features: int,
               gate_initial_value: float=0.,
               temperature: float=1.):
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)
    self.activation = nn.ReLU()
    self.gate_initial_value = gate_initial_value
    self.temperature = temperature

    self.gatefn = GateFn(temperature=temperature)
    self.strengths = nn.Parameter(gate_initial_value * torch.ones(out_features), requires_grad=True)

  def _gate(self, x: Tensor) -> Tensor:
    return self.gatefn(x, self.strengths)

  def forward(self, x: Tensor) -> Tensor:
    x = self.linear(x)
    x = self.activation(x)
    x = self._gate(x)
    return x

  def gate_values(self) -> Tensor:
    return self.gatefn.gates(self.strengths)
  
  def sigmoid_values(self) -> Tensor:
    return torch.sigmoid(self.strengths)

  def arch_parameters(self) -> nn.Parameter:
    return self.strengths

  def weight_parameters(self) -> List[nn.Parameter]:
    params = []
    for name, param in self.named_parameters():
      if 'strengths' not in name:
        params.append(param)
    return params
