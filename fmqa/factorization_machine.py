"""
Factorization Machine implemented with PyTorch
"""

import numpy as np
import torch
import torch.nn as nn


def triu_mask(input_size: int, device=None, dtype=None):
    idx = torch.arange(input_size, device=device)
    mask = (idx.view(-1, 1) < idx.view(1, -1)).to(dtype=dtype if dtype is not None else torch.float32)
    return mask


def VtoQ(V: torch.Tensor) -> torch.Tensor:
    Q = V.transpose(0, 1) @ V
    mask = triu_mask(Q.shape[0], device=Q.device, dtype=Q.dtype)
    return Q * mask


class QuadraticLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self._optimizer = None

    def init_params(self):
        with torch.no_grad():
            for p in self.parameters():
                nn.init.normal_(p, mean=0.0, std=1.0)

    def train_model(self, x, y, num_epoch=100, learning_rate=1.0e-2, device=None):
        # device が指定されていたら、モデル本体もそこへ移動
        if device is not None:
            self.to(device)

        self.train()

        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        y = torch.as_tensor(y, dtype=torch.float32, device=device).view(-1)

        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        else:
            for g in self._optimizer.param_groups:
                g["lr"] = learning_rate

        for _ in range(num_epoch):
            self._optimizer.zero_grad(set_to_none=True)
            output = self(x).view(-1)
            loss = torch.mean((y - output) ** 2)
            loss.backward()
            self._optimizer.step()

    def get_bhQ(self):
        raise NotImplementedError()


class FactorizationMachine(QuadraticLayer):
    def __init__(self, input_size, factorization_size=8, act="identity"):
        super().__init__()
        self.factorization_size = int(factorization_size)
        self.input_size = int(input_size)

        self.h = nn.Parameter(torch.empty(self.input_size, dtype=torch.float32))
        if self.factorization_size > 0:
            self.V = nn.Parameter(torch.empty(self.factorization_size, self.input_size, dtype=torch.float32))
        else:
            self.V = nn.Parameter(torch.empty(1, self.input_size, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.act = act
        self.init_params()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.view(1, -1)

        if self.factorization_size <= 0:
            return (self.bias + x @ self.h).view(-1)

        Q = VtoQ(self.V)
        Qx = x @ Q.transpose(0, 1)

        if self.act == "identity":
            out = self.bias + (x @ self.h) + torch.sum(x * Qx, dim=1)
            return out.view(-1)
        if self.act == "sigmoid":
            out = self.bias + (x @ self.h) + torch.sum(x * Qx, dim=1)
            return torch.sigmoid(out).view(-1)
        if self.act == "tanh":
            out = self.bias + (x @ self.h) + torch.sum(x * Qx, dim=1)
            return torch.tanh(out).view(-1)

        raise ValueError(f"Unknown activation: {self.act}")

    def get_bhQ(self):
        with torch.no_grad():
            V = torch.zeros_like(self.V) if self.factorization_size == 0 else self.V
            Q = VtoQ(V)
            b = float(self.bias.detach().cpu().view(-1)[0].item())
            h = self.h.detach().cpu().numpy()
            Qn = Q.detach().cpu().numpy()
        return b, h, Qn
