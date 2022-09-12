from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Type, cast, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

T: TypeAlias = NDArray[np.floating[Any]]
F: TypeAlias = float | T  # float | NDArray[np.floating[Any]]


class NToOneModel(metaclass=ABCMeta):
    ERRMSG: str = \
        "Cannot adjust the variance, call this method with adjust=True first."

    @abstractmethod
    def __init__(
            self, seed: int, n_in: int,
            tgt_snratio: float = 1., tgt_var: float = 1., tgt_mean: float = 0.
            ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def __call__(
            self, x: F, e: F,
            adjust: bool = False
            ) -> F:
        raise NotImplementedError()


class NoiseOnly(NToOneModel):
    def __init__(
            self, seed: int = 0, n_in: int = 1,
            tgt_snratio: float = 1.0, tgt_var: float = 1., tgt_mean: float = 0.
            ) -> None:
        """Returns the standardized input noise.

        Parameters
        ----------
        seed : int, optional
            Ignored, by default 0
        n_in : int, optional
            Ignored, by default 1
        tgt_snratio : float, optional
            Ignored, by default 1.0
        tgt_var : float, optional
            The target variance of the input noise, by default 1.
        tgt_mean : float, optional
            The target mean of the input noise, by default 0.
        """
        self._var = tgt_var
        self._m = tgt_mean
        self._gamma_e: float | None = None
        self._mu_e: F | None = None

    def __call__(
            self, x: F, e: F,
            adjust: bool = False
            ) -> F:
        if adjust:
            self._gamma_e = np.sqrt(self._var / np.var(e))
            self._mu_e = self._m - self._gamma_e * np.mean(e)
        elif self._gamma_e is None:
            raise ValueError(super().ERRMSG)

        return cast(F, self._gamma_e) * e + cast(F, self._mu_e)  # noise

    def __str__(self) -> str:
        return "NoiseOnlyModel"


class GaussianProcesses(object):
    def __init__(self, m: T, w1: T, w2: T, s: float) -> None:
        """A weighted sum of Gaussian processes.

        Parameters
        ----------
        m : NDArray[np.floating[Any]]
            Mean vectors of Gaussian processes with shape=(n, q), where n is
            the dimension of the input variable and q is the number of Gaussian
            kernels in one GP.
        w1 : NDArray[np.floating[Any]]
            Weights of GPs with shape=(n, q).
        w2 : NDArray[np.floating[Any]]
            Weights for aggregating GPs with shape=(n,).
        s : float
            The standard deviation of GPs. Note that all GPS has the same std.
        """
        super().__init__()
        # check shapes
        if (m.ndim != 2) or (w1.ndim != 2) or (w2.ndim != 1):
            raise ValueError("ndim is incorrect.")
        if (m.shape != w1.shape) or (m.shape[0] != w2.shape[0]):
            raise ValueError("shape is incorrect.")
        # add first axis corresponding to the number of samples.
        self._m = m[None]
        self._w1 = w1[None]
        self._w2 = w2[None]
        self._s = s

    def __call__(self, x: F) -> F:
        """

        Parameters
        ----------
        x : float | NDArray[np.floating[Any]]
            Samples. If x is a float, it will be regarded as a sample of 1D
            variable. If x is a 1D array of length N, regarded as N samples of
            1D variable. If x is a 2D array of shape=(N,M), regarded as N
            samples of MD variable.

        Returns
        -------
        float | NDArray[np.floating[Any]]
        """
        # Adjust size to (#samples, #dimensions, #kernels)
        if not hasattr(x, "ndim"):
            _x = np.array([x])[None, None]
        else:
            x = cast(T, x)
            if x.ndim == 1:
                _x = x[:, None, None]
            elif x.ndim == 2:
                _x = x[:, :, None]
            else:
                raise ValueError(f"Input dimension is incorrect: {x.ndim}.")

        # Gaussian kernel values.
        gks = np.exp(- (_x - self._m)**2 / self._s**2)
        # GPs, weighted sum of q Gaussian kernels.
        gps = (gks * self._w1).sum(axis=-1)
        # Weighted sum of GPs.
        z = (gps * self._w2).sum(axis=-1)

        return z if hasattr(x, "ndim") else float(z)


class Sigmoidals(object):
    def __init__(self, m: T, w: T, a: float) -> None:
        """A weighted sum of sigmoidal functions.

        Parameters
        ----------
        m : NDArray[np.floating[Any]]
            A mean vector of sigmoidal functions with shape=(q,), where q is
            the number of sigmoidal functions.
        w : NDArray[np.floating[Any]]
            Weights for the linear aggregation of the functions with
            shape=(q,).
        a : float
            A scale of sigmoidal functions. Note that all the functions use
            the same value.
        """
        super().__init__()
        # Check shapes
        if (m.ndim != 1) or (w.ndim != 1):
            raise ValueError("Dimension is incorrect.")
        if m.shape != w.shape:
            raise ValueError("Shapes are incorrect.")
        # add first dimension corresponding to the number of samples.
        self._m = m[None]
        self._w = w[None]
        self._a = a

    def __call__(self, z: F) -> F:
        """_summary_

        Parameters
        ----------
        z : float | NDArray[np.floating[Any]]
            Samples of an input variable. If z is a float, it will be regarded
            as a sample. If z is a 1D array of length N, regarded as N samples.

        Returns
        -------
        float | NDArray[np.floating[Any]]
        """
        # Adjust the shape to (#samlpes, #kernels)
        if not hasattr(z, "ndim"):
            _z = np.array([z])[None]
        else:
            z = cast(T, z)
            if z.ndim == 1:
                _z = z[:, None]
            else:
                raise ValueError(f"Input dimension is incorrect: {z.ndim}.")

        # Sigmoidal function values.
        ss = 1. / (1. + np.exp(-self._a * (_z - self._m)))
        # Weighted sum of sigmoidal functions
        y = (ss * self._w).sum(axis=-1)

        return y if hasattr(z, "ndim") else float(y)


class PostNonlinearNToOne(NToOneModel):
    def __init__(
            self, seed: int, n_in: int,
            n_f1: int = 5, n_f2: int = 5,
            tgt_snratio: float = 1., tgt_var: float = 1., tgt_mean: float = 0.
            ) -> None:
        """Generate an N-to-one post nonlinear model.

        Parameters
        ----------
        seed : int
            Random seed
        n_in : int
            Input dimension.
        n_f1 : int, optional
            The number of Gaussian processes in f1, by default 5
        n_f2 : int, optional
            The number of sigmoidal functions in f2, by default 5
        tgt_snratio : float, optional
            Target value of SN ratio, by default 1.
        tgt_var : float, optional
            Target variance of the output, by default 1.
        tgt_mean : float, optional
            Target mean of the output, by default 0.
        """
        self._rng = np.random.RandomState(seed)
        self._q1 = n_f1
        self._q2 = n_f2
        self._n = n_in

        self._var = tgt_var
        self._m = tgt_mean
        self._gamma_e: float | None = None
        self._gamma_1: float | None = None
        self._gamma_2: float | None = None
        self._mu_1: float | None = None
        self._mu_2: F | None = None
        self._lambda = tgt_snratio / (1. + tgt_snratio)

        self.generate_f1()
        self.generate_f2()

    def generate_f1(self) -> None:
        r = self._rng
        # Mean values of all Gaussians.
        self._f1_m = r.random((self._n, self._q1)) * 2. - 0.5
        # Weights in GPs.
        self._f1_w1 = r.random((self._n, self._q1))
        self._f1_w1 /= self._f1_w1.sum(axis=-1, keepdims=True)
        # Weights for a linear aggrigation of GPs.
        self._f1_w2 = r.random((self._n,))
        self._f1_w2 /= self._f1_w2.sum()
        # Standard deviation of GPs
        self._f1_s = 0.3

        self.f1 = GaussianProcesses(self._f1_m, self._f1_w1,
                                    self._f1_w2, self._f1_s)

    def generate_f2(self) -> None:
        r = self._rng
        # Mean values of sigmoidal functions.
        self._f2_m = r.random((self._q2,)) * 1.
        # Weights in the linear aggrigation of sigmoidal functions.
        self._f2_w = r.random((self._q2,))
        self._f2_w /= self._f2_w.sum()
        # scale
        self._f2_a = 10.0

        self.f2 = Sigmoidals(self._f2_m, self._f2_w, self._f2_a)

    def __call__(self, x: F, e: F, adjust: bool = False) -> F:
        if (not adjust) and ((self._gamma_e is None) or
                             (self._gamma_1 is None) or
                             (self._gamma_2 is None) or
                             (self._mu_1 is None) or
                             (self._mu_2 is None)):
            raise ValueError(super().ERRMSG)
        # Nonlinear transformation.
        z = self.f1(x)
        # Noise addition.
        if adjust:
            # variances
            self._gamma_e = np.sqrt(self._var / np.var(e))
            self._gamma_1 = np.sqrt(self._var / np.var(z))
        z = cast(float, self._gamma_1) * z
        e = cast(float, self._gamma_e) * e
        z = np.sqrt(self._lambda) * z + np.sqrt(1. - self._lambda) * e
        if adjust:
            # means
            self._mu_1 = self._m - np.mean(z)
        z = z + cast(float, self._mu_1)

        # Nonlinear invertible transformation.
        y = self.f2(z)

        # Adjust the variance of y
        if adjust:
            self._gamma_2 = np.sqrt(self._var / np.var(y))
        y = cast(float, self._gamma_2) * y
        # Adjust the mean of y
        if adjust:
            self._mu_2 = self._m - np.mean(y)
        y = y + cast(float, self._mu_2)

        return y

    def __str__(self) -> str:
        return f"{self._n}to1_PNLModel"


class LinearNToOne(NToOneModel):
    def __init__(
            self, seed: int, n_in: int,
            tgt_snratio: float = 1., tgt_var: float = 1., tgt_mean: float = 0.
            ) -> None:
        """Generate a linear aggregation function.

        Parameters
        ----------
        seed : int
            Random seed.
        n_in : int
            Input dimension.
        tgt_snratio : float, optional
            Target SN ratio in terms of teh variance, by default 1.
        tgt_var : float, optional
            Target variance of the output, by default 1.
        tgt_mean : float, optional
            Target mean of the output, by default 0.
        """
        self._rng = np.random.RandomState(seed)
        self._n = n_in

        self._var = tgt_var
        self._lambda = tgt_snratio / (tgt_snratio + 1.)
        self._gamma_x: float | None = None
        self._gamma_e: float | None = None

        self.generate_coefficients()

    def generate_coefficients(self) -> None:
        # ~U(-1, 1)
        self.a = 2. * self._rng.rand(self._n) - 1.

    def __call__(
            self, x: F, e: F, adjust: bool = False) -> F:
        if (not adjust) and ((self._gamma_x is None) or
                             (self._gamma_e is None)):
            raise ValueError(super().ERRMSG)

        x = np.dot(x, self.a)
        if adjust:
            self._gamma_x = np.sqrt(self._var / np.var(x))
            self._gamma_e = np.sqrt(self._var / np.var(e))

        x = cast(float, self._gamma_x) * x
        e = cast(float, self._gamma_e) * e
        # for debug
        self._last_x = x
        self._last_e = e

        y = np.sqrt(self._lambda) * x + np.sqrt(1. - self._lambda) * e
        return cast(F, y)

    def __str__(self) -> str:
        return f"{self._n}to1_LinearModel"


class AutoGenCausalStructure(object):
    """ Generate a causal structure.
    Based on N->1 substructures.
    """
    def __init__(
            self, seed: int, n_variables: int,
            model: Type[NToOneModel] = PostNonlinearNToOne,
            allow_multi_DAGs: bool = False,
            **n_to_one_kwargs: Any
            ) -> None:
        """Generate a causal DAG and equations.

        Parameters
        ----------
        seed : int
            Random seed.
        n_variables : int
            The number of variables.
        model : Type[NToOneModel], optional
            Base N-to-one class, by default PostNonlinearNToOne
        allow_multi_DAGs : bool, optional
            If False, all nodes will be connected, by default False
        """
        super().__init__()

        self._p = n_variables
        self._rng = np.random.RandomState(seed)
        self._model = model
        # Other kwargs for NtoOneModel
        self.n_to_one_kwargs = n_to_one_kwargs
        # Generate a DAG
        self._gen_DAG()
        while (not allow_multi_DAGs) and (not self._nodes_connected()):
            # Repeat generation
            self._gen_DAG()

        # Generate all N->1 substructures.
        self._gen_substructures()

    def _gen_DAG(self) -> None:
        self.causal_order = self._rng.permutation(self._p)

        # Each edge is generated with probability 2/(n-1)
        a = self._rng.random((self._p, self._p)) < 2./(self._p-1)
        a = np.tril(a.astype(int), k=-1)

        idx = np.argsort(self.causal_order)
        # If the (i,j)-element has a non-zero value, there is a edge xj -> xi.
        self.adj_mat = a[idx][:, idx]

    def _nodes_connected(self) -> bool:
        """ Check if all nodes are connected.
        """
        # naive implementation (depth-first search)
        a = self.adj_mat + self.adj_mat.T  # bidirectional graph

        def add_parents(i: int, s: set[int]) -> set[int]:
            parents = np.argwhere(a[i]).flatten()  # parents of i
            for p in parents:
                if p not in s:
                    s.add(p)
                    s = add_parents(p, s)
            return s
        # Check if the number of reachable nodes from node 0 equals to the
        # number of nodes.
        return len(add_parents(0, set([0]))) == self._p

    def _gen_substructures(self) -> None:
        """
        Generate all n to one substructures by extracting them from the DAG.
        """
        m: list[NToOneModel] = []  # N-to-one models.
        # Non-zero elements in i-th row are the parents of xi.
        for parents in [np.argwhere(_).flatten() for _ in self.adj_mat]:
            if len(parents) == 0:  # If the node is a source node.
                m.append(NoiseOnly())
            else:  # If it has a parent.
                m.append(
                    self._model(seed=self._rng.randint(1000000),
                                n_in=len(parents), **self.n_to_one_kwargs))
        self.models = m

    def generate_samples(self, e: T, adjust: bool = False) -> T:
        """Generate samples from given noises.

        Parameters
        ----------
        e : NDArray[np.floating[Any]]
            Noises. If e is a 1D array with shape=(N,), it will be regarded as
            one sample of N variables. If 2D array with shape=(M,N), regarded
            as M samples of N variables.
        adjust : bool, optional
            If True, adjust SN ratios, by default False

        Returns
        -------
        NDArray[np.floating[Any]]
            Samples.
        """
        dim1 = (e.ndim == 1)
        if dim1:
            e = e[None]
        x = np.zeros_like(e) * np.nan

        # Generate samples in the causal order.
        for i in self.causal_order:
            p = np.argwhere(self.adj_mat[i]).flatten()  # parents of xi
            x[:, i] = self.models[i](x[:, p], e[:, i], adjust=adjust)

        return x.astype("float32")


def generate_samples(
        seed: int, num_variables: int, num_samples: int,
        seed_noise: int | None = None,
        n_to_one: Type[NToOneModel] = PostNonlinearNToOne,
        **n_to_one_kwargs: Any
        ) -> tuple[T, AutoGenCausalStructure]:
    """Generate samples.

    Parameters
    ----------
    seed : int
        Random seed to generate a model.
    num_variables : int
        The number of variables.
    num_samples : int
        The number of samples.
    seed_noise : int | None, optional
        Random seed to generate noise variables. If None, use `seed` instead,
        by default None
    n_to_one : Type[NToOneModel], optional
        N-to-one model class, by default PostNonlinearNToOne

    Returns
    -------
    NDArray[np.floating[Any]]
        Samples
    """
    model = AutoGenCausalStructure(
        seed, num_variables, n_to_one,
        allow_multi_DAGs=False, **n_to_one_kwargs
    )

    # Adjust internal parameters by estimating variances with 100 samples.
    n_adjust = 100
    e = np.random.RandomState(123).random((n_adjust, num_variables)) * 1.0
    _ = model.generate_samples(e, adjust=True)

    # Generate noises
    rng = np.random.RandomState(seed if seed_noise is None else seed_noise)
    e = rng.random((num_samples, num_variables)) * 1.0  # ~U(0, 1)

    # Generate samples
    x = model.generate_samples(e)

    return x, model
