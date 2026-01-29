import numpy as np
import torch
from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.vartypes import Vartype

from .factorization_machine import FactorizationMachine


class FactorizationMachineBinaryQuadraticModel(BinaryQuadraticModel):
    def __init__(self, input_size, vartype, act="identity", **kwargs):
        init_linear = {i: 0.0 for i in range(input_size)}
        init_quadratic = {}
        init_offset = 0.0
        super().__init__(init_linear, init_quadratic, init_offset, vartype, **kwargs)
        self.fm = FactorizationMachine(input_size, act=act)

    @classmethod
    def from_data(cls, x, y, act="identity", num_epoch=1000, learning_rate=1.0e-2, device=None, **kwargs):
        if np.all((x == 0) | (x == 1)):
            vartype = Vartype.BINARY
        elif np.all((x == -1) | (x == 1)):
            vartype = Vartype.SPIN
        else:
            raise ValueError("input data should BINARY or SPIN vectors")

        input_size = x.shape[-1]
        fmbqm = cls(input_size, vartype, act, **kwargs)
        fmbqm.train(x, y, num_epoch, learning_rate, init=True, device=device)
        return fmbqm

    def train(self, x, y, num_epoch=1000, learning_rate=1.0e-2, init=False, device=None):
        if init:
            self.fm.init_params()

        self._check_vartype(x)
        self.fm.train_model(x, y, num_epoch, learning_rate, device=device)

        if self.vartype == Vartype.SPIN:
            h, J, b = self._fm_to_ising()
            self.offset = b
            for i in range(self.fm.input_size):
                self.linear[i] = h[i]
                for j in range(i + 1, self.fm.input_size):
                    self.quadratic[(i, j)] = J.get((i, j), 0)
        elif self.vartype == Vartype.BINARY:
            Q, b = self._fm_to_qubo()
            self.offset = b
            for i in range(self.fm.input_size):
                self.linear[i] = Q[(i, i)]
                for j in range(i + 1, self.fm.input_size):
                    self.quadratic[(i, j)] = Q.get((i, j), 0)

    def predict(self, x, device=None):
        self._check_vartype(x)

        # deviceが未指定なら、モデルのdeviceに合わせる（混在エラー回避）
        if device is None:
            device = next(self.fm.parameters()).device

        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = self.fm(x_t).detach().cpu().numpy()
        return pred

    def _check_vartype(self, x):
        if (self.vartype is Vartype.BINARY) and np.all((1 == x) | (0 == x)) or \
           (self.vartype is Vartype.SPIN) and np.all((1 == x) | (-1 == x)):
            return
        raise ValueError("input data should be of type", self.vartype)

    def _fm_to_ising(self, scaling=True):
        b, h, J = self.fm.get_bhQ()
        if self.vartype is Vartype.BINARY:
            b = b + np.sum(h) / 2 + np.sum(J) / 4
            h = (2 * h + np.sum(J, axis=0) + np.sum(J, axis=1)) / 4.0
            J = J / 4.0
        if scaling:
            scaling_factor = max(np.max(np.abs(h)), np.max(np.abs(J)))
            if scaling_factor != 0:
                b /= scaling_factor
                h /= scaling_factor
                J /= scaling_factor
        return {key: h[key] for key in range(len(h))}, {key: J[key] for key in zip(*J.nonzero())}, b

    def _fm_to_qubo(self, scaling=True):
        b, h, Q = self.fm.get_bhQ()
        if self.vartype is Vartype.SPIN:
            b = b - np.sum(h) + np.sum(Q)
            h = 2 * (h - np.sum(Q, axis=0) - np.sum(Q, axis=1))
            Q = 4 * Q
        Q[np.diag_indices(len(Q))] = h
        if scaling:
            scaling_factor = np.max(np.abs(Q))
            if scaling_factor != 0:
                b /= scaling_factor
                Q /= scaling_factor

        Q_dict = {key: Q[key] for key in zip(*Q.nonzero())}
        for i in range(len(Q)):
            Q_dict[(i, i)] = Q[i, i]
        return Q_dict, b
