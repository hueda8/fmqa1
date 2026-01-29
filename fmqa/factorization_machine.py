"""
Factorization Machine implemented with PyTorch
"""

__all__ = [
    "FactorizationMachineBinaryQuadraticModel", "FMBQM"
]

import numpy as np
import torch
import torch.nn as nn


def triu_mask(input_size: int, device=None, dtype=None):
    """Generate a square matrix with its upper triangular elements (i<j) being 1 and others 0."""
    idx = torch.arange(input_size, device=device)
    # mask[i, j] = 1 if i < j else 0
    mask = (idx.view(-1, 1) < idx.view(1, -1)).to(dtype=dtype if dtype is not None else torch.float32)
    return mask


def VtoQ(V: torch.Tensor) -> torch.Tensor:
    """Calculate interaction strength by inner product of feature vectors.

    V: (k, d)
    Returns Q: (d, d) with upper-triangular (i<j) entries kept, others zeroed.
    """
    # Q = V^T V
    Q = V.transpose(0, 1) @ V  # (d, d)
    mask = triu_mask(Q.shape[0], device=Q.device, dtype=Q.dtype)
    return Q * mask


class QuadraticLayer(nn.Module):
    """A neural network layer which applies quadratic function on the input.

    This class defines train_model() method for easy use.
    """

    def __init__(self):
        super().__init__()
        self._optimizer = None

    def init_params(self):
        """Initialize parameters similar to mx.init.Normal()."""
        # PyTorch標準の初期化（正規分布）に寄せる
        for m in self.modules():
            if isinstance(m, nn.Parameter):
                continue
        for p in self.parameters():
            if p.dim() >= 1:
                nn.init.normal_(p, mean=0.0, std=1.0)

    def train_model(self, x, y, num_epoch=100, learning_rate=1.0e-2, device=None):
        """Training of the regression model using Adam optimizer."""
        self.train()

        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        y = torch.as_tensor(y, dtype=torch.float32, device=device)

        if y.ndim == 1:
            y = y.view(-1)

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
    """Factorization Machine as a neural network layer.

    Args:
        input_size (int):
            The dimension of input value.
        factorization_size (int (<=input_size)):
            The rank of decomposition of interaction terms.
        act (string, optional):
            "identity", "sigmoid", or "tanh". (default="identity")
    """

    def __init__(self, input_size, factorization_size=8, act="identity"):
        super().__init__()
        self.factorization_size = int(factorization_size)
        self.input_size = int(input_size)

        # MXNet版: h shape=(d,), V shape=(k,d), bias shape=(1,)
        self.h = nn.Parameter(torch.empty(self.input_size, dtype=torch.float32))
        if self.factorization_size > 0:
            self.V = nn.Parameter(torch.empty(self.factorization_size, self.input_size, dtype=torch.float32))
        else:
            # dummy V
            self.V = nn.Parameter(torch.empty(1, self.input_size, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.act = act
        self.init_params()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, d)
        Returns: (N,)
        """
        if x.ndim == 1:
            x = x.view(1, -1)

        if self.factorization_size <= 0:
            return (self.bias + x @ self.h).view(-1)

        Q = VtoQ(self.V)  # (d, d)
        # equivalent of MXNet FullyConnected with weight=Q, no bias: x @ Q.T
        Qx = x @ Q.transpose(0, 1)  # (N, d)

        if self.act == "identity":
            act_fn = lambda t: t
        elif self.act == "sigmoid":
            act_fn = torch.sigmoid
        elif self.act == "tanh":
            act_fn = torch.tanh
        else:
            raise ValueError(f"Unknown activation: {self.act}")

        out = self.bias + (x @ self.h) + torch.sum(x * Qx, dim=1)
        return act_fn(out).view(-1)

    def get_bhQ(self):
        """Returns (bias, h, Q) as (float, np.ndarray, np.ndarray)."""
        with torch.no_grad():
            if self.factorization_size == 0:
                V = torch.zeros_like(self.V)
            else:
                V = self.V
            Q = VtoQ(V)
            b = float(self.bias.detach().cpu().view(-1)[0].item())
            h = self.h.detach().cpu().numpy()
            Qn = Q.detach().cpu().numpy()
        return b, h, Qn
