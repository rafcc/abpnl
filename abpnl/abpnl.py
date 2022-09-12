from __future__ import annotations

from collections.abc import Callable, Sequence, Iterable
from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import NDArray, NBitBase
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import Dataset

T = TypeVar('T', bound=np.floating[NBitBase])


def gram_Gaussian(x: Tensor, sigma: float = 1.) -> Tensor:
    """Gram matrix with Gaussian kernel.

    Parameters
    ----------
    x : Tensor
        Samples.
    sigma : float, optional
        Bandwidth of the Gaussian kernel, by default 1.

    Returns
    -------
    Tensor
        Gram matrix.
    """
    x = x[:, None]
    d = torch.abs(x - x.t())
    return torch.exp(-d**2 / sigma**2)


def standardize(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Standardize samples over the first dimension.

    Parameters
    ----------
    x : Tensor
        Samples. x.shape = (#samples, #variables)
    eps : float, optional
        epsilon, by default 1e-5

    Returns
    -------
    Tensor
        Standardized samples.
    """
    return (x - x.mean(0)) / (x.std(0, unbiased=False) + eps)


def max_Gaussian_eHSIC(x1s: Tensor, x2s: Tensor) -> Tensor:
    """Maximum empirical HSIC.

    Parameters
    ----------
    x1s : Tensor
        Samples. x1.shape = (#samples, #variables).
    x2s : Tensor
        Samples. x2.shape = (#samples, #variables).

    Returns
    -------
    Tensor
        _description_
    """
    n = len(x1s)
    ehsic: Tensor | None = None

    for x1 in standardize(x1s).t():
        g1 = gram_Gaussian(x1)
        for x2 in standardize(x2s).t():
            g2 = gram_Gaussian(x2)  # TODO Reusable.

            h = torch.eye(n, device=x1.device) - 1./n
            ehsic_new = 1./(n**2) * \
                torch.sum(torch.mm(torch.mm(h, g1), h).t() * g2)

            if (ehsic is None) or (ehsic_new > ehsic):
                ehsic = ehsic_new
    return cast(Tensor, ehsic)


class MLPN(nn.Module):
    def __init__(
            self, nin: int, nout: int, nunits: int, nlayers: int,
            actf: Callable[[Tensor], Tensor] = F.leaky_relu,
            dropout: bool = False
            ) -> None:
        """Multi-layer perceptron.

        Parameters
        ----------
        nin : int
            Input dimension.
        nout : int
            Output dimension.
        nunits : int
            The number of units in each layer.
        nlayers : int
            The number of layers.
        actf : Callable[[Tensor], Tensor], optional
            Activation function, by default `torch.nn.functional.leaky_relu`.
        dropout : bool
            If True, use Dropout for regularization.
        """
        super().__init__()
        for i in range(nlayers):
            setattr(
                self, f"fc{i}",
                nn.Linear(nin if i == 0 else nunits,
                          nout if i == nlayers-1 else nunits)
                )
        self._dropout = dropout
        if self._dropout:
            for i in range(nlayers-1):
                setattr(
                    self, f"dr{i}",
                    nn.Dropout(0.1)
                )
        self._nlayers = nlayers
        self._actf = actf

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self._nlayers-1):
            x = self._actf(getattr(self, f"fc{i}")(x))
            if self._dropout:
                x = getattr(self, f"dr{i}")(x)
        return cast(Tensor, getattr(self, f"fc{self._nlayers-1}")(x))


class AbPNLModel(nn.Module):

    def __init__(
            self, nx: int, ny: int = 1, nz: int = 5, nl: int = 5,
            actf: Callable[[Tensor], Tensor] = F.leaky_relu, a: float = .5,
            lossf_ind: Callable[[Tensor, Tensor], Tensor] = max_Gaussian_eHSIC,
            lossf_inv: Callable[[Tensor, Tensor], Tensor] = nn.MSELoss(),
            dropout: bool = False
            ) -> None:
        """Network model for AbPNL.

        Parameters
        ----------
        nx : int
            The number of causes.
        ny : int, optional
            The number of effects, by default 1.
        nz : int, optional
            The number of units in each fully-connected layer, by default 5.
        nl : int, optional
            The number of layers in each MLPs, by default 5
        actf : Callable[[Tensor], Tensor], optional
            Activation function, by default `torch.nn.functional.leaky_relu`.
        a : float, optional
            Relative weight of independence loss against invertibility loss,
            by default .5.
        lossf_ind : Callable[[Tensor, Tensor], Tensor], optional
            Independence loss function, by default max_Gaussian_eHSIC
        lossf_inv : Callable[[Tensor, Tensor], Tensor], optional
            Invertibility loss function, by default nn.MSELoss()
        dropout : bool
            If True, use dropout for regularization.
        """
        super().__init__()

        def _Model(ni: int, no: int) -> MLPN:
            return MLPN(ni, no, nz, nl, actf, dropout)
        self.f1 = _Model(nx, 1)
        self.f2 = _Model(1, ny)
        self.f2i = _Model(ny, 1)

        self._a = a
        self.lossf_ind = lossf_ind
        self.lossf_inv = lossf_inv

        # for logging
        self._loss_indep: float
        self._loss_inver: float
        self._loss: float
        self._e: NDArray[Any]
        self._y_: NDArray[Any]

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward calculation.

        Parameters
        ----------
        x : Tensor
            Batch of cause variables. The first axis corresponds to batch.
        y : Tensor
            Batch of effect variables. The first axis corresponds to batch.

        Returns
        -------
        Tensor
            Loss.
        """
        y = y[:, None]

        z = self.f1(x)
        w = self.f2i(y)
        y_ = self.f2(w)
        e = w - z

        self.last_z = z  # z = f1(x)
        self._e = e.detach().numpy()  # estimated errors.
        self._y_ = y_.detach().numpy()  # reconstructed effects.

        loss_indep = self.lossf_ind(x, e)
        loss_inver = self.lossf_inv(y, y_)
        loss = self._a * loss_indep + (1. - self._a) * loss_inver

        # for logging
        self._loss_indep = loss_indep.item()
        self._loss_inver = loss_inver.item()
        self._loss = loss.item()

        return loss

    @property
    def loss_ind(self) -> float:
        return self._loss_indep

    @property
    def loss_inv(self) -> float:
        return self._loss_inver

    @property
    def loss(self) -> float:
        return self._loss

    @property
    def e(self) -> NDArray[Any]:
        return self._e

    @property
    def y_(self) -> NDArray[Any]:
        return self._y_


class NtoOneDataset(Dataset[tuple[Sequence[T], T]]):
    def __init__(
            self, dataarray: Sequence[Sequence[T]],
            causes: Iterable[int], effect: int
            ) -> None:
        """Dataset with N causes and one effect variables.

        Parameters
        ----------
        dataarray : Sequence[Sequence[T]]
            Data array with shape = (#samlpes, #variables).
        causes : Iterable[int]
            Indices of cause variables.
        effect : int
            An index of an effect variable.
        """
        super().__init__()

        self._data = np.array(dataarray)
        self._causes_d = tuple(causes)
        self._effect_d = effect
        self.sef_causes_effect()

    def _check_indices(self) -> None:
        if len(self._causes_d) == 0:
            raise ValueError("No causes.")
        elif (min(self._causes_d) < 0) or (self._effect_d < 0):
            raise ValueError("Indices must be positive values.")
        elif self._effect_d in self._causes_d:
            raise ValueError("Indices overlap.")
        elif max(max(self._causes_d), self._effect_d) >= len(self._data):
            raise ValueError("Indices out of range.")

    def set_causes(
            self, causes: Iterable[int] | None = None, check: bool = True
            ) -> None:
        self._causes = tuple(self._causes_d if causes is None else causes)
        if check:
            self._check_indices()

    def set_effect(
            self, effect: int | None = None, check: bool = True
            ) -> None:
        self._effect = self._effect_d if effect is None else effect
        if check:
            self._check_indices()

    def sef_causes_effect(
            self, causes: Iterable[int] | None = None,
            effect: int | None = None, check: bool = True
            ) -> None:
        self.set_causes(causes, False)
        self.set_effect(effect, False)
        if check:
            self._check_indices()

    @property
    def causes(self) -> tuple[int, ...]:
        return tuple(_ for _ in self._causes)

    @property
    def effect(self) -> int:
        return self._effect

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[Sequence[T], T]:
        xc = self._data[idx, self._causes]
        xe = self._data[idx, self._effect]
        return xc, xe
