from __future__ import annotations

from typing import TypeVar, cast, Any
import numpy as np
from numpy.typing import NDArray
import scipy.stats

T = TypeVar('T', bound=np.floating[Any])


def standardize(x: NDArray[T], eps: float = 1e-5) -> NDArray[T]:
    return cast(NDArray[T], (x - x.mean(0)) / (x.std(0, ddof=0) + eps))


def g(
        x: NDArray[T], bandwidth: float | None = None,
        n: int = 200, shuffle: bool = True
        ) -> NDArray[T]:
    """Gram matrix with the Gaussian kernel.

    Parameters
    ----------
    x : NDArray[T]
        Samples with shape=(#samples).
    bandwidth : Optional[float], optional
        A bandwidth of the Gaussian kernel. If None, heuristics will be used to
        decide the bandwidth, by default None.
    n : int, optional
        The number of used samples in the bandwidth heuristics, by default 200.
        If n<=0, use all. Only works when bandwidth=None.
    shuffle : bool, optional
        If True, samples will be shuffled for the bandwidth heuristics, by
        default True. Only works when bandwidth=None.

    Returns
    -------
    NDArray[T]
        Gram matrix.
    """
    if bandwidth is not None:
        assert bandwidth > 0.
        d = np.abs(x - x.T)
        s = bandwidth
    else:
        x_o = x
        if n > 0:
            if shuffle:
                rng = np.random.RandomState(12345679)
                x_ = rng.permutation(x)[:n]
            else:
                x_ = x[:n]
            x_o = x
            x = x_
        d = np.abs(x - x.T)
        s = np.median(d[d > 0])  # TODO When |unique(x)|=1, it collapses.
        if n > 0:
            d = np.abs(x_o - x_o.T)
    return cast(NDArray[T], np.exp(-d**2 / (s**2)))


def test_th(
        x: NDArray[T], y: NDArray[T], a: float = 0.05,
        bandwidth: float | None = None,
        n: int = 200, shuffle: bool = True
        ) -> tuple[float, float, float, float, float, float]:
    """ Calc threshold for HSIC independence test.
    x,y: samples to be tested.
    a  : test threshold in [0,1].

    Returns
    th     : test threshold. If m*HSIC is smaller than it, pass the test.
    varHSIC: intermediate value.
    mHSIC  : intermediate value.
    alpha  :
    beta   :
    stats  :
    """
    if x.ndim < 2:
        x = x[:, None]
    if y.ndim < 2:
        y = y[:, None]

    n = x.shape[0]
    h = np.eye(n) - 1./n
    k = g(x, bandwidth=bandwidth, n=n, shuffle=shuffle)
    l = g(y, bandwidth=bandwidth, n=n, shuffle=shuffle)
    kc = np.dot(np.dot(h, k), h)
    lc = np.dot(np.dot(h, l), h)
    # variance
    varHSIC = (kc * lc)**2
    varHSIC = 1./n/(n-1) * (np.sum(varHSIC) - np.sum(np.diag(varHSIC)))
    varHSIC = 2*(n-4)*(n-5)/n/(n-1)/(n-2)/(n-3) * varHSIC
    # mean
    mX2 = 1./n/(n-1) * (np.sum(k) - np.sum(np.diag(k)))
    mY2 = 1./n/(n-1) * (np.sum(l) - np.sum(np.diag(l)))
    mHSIC = 1./n * (1 + mX2*mY2 - mX2 - mY2)
    # params of Gamma dist.
    alp = mHSIC**2 / varHSIC
    bet = n * varHSIC / mHSIC
    # threshold
    th = scipy.stats.gamma.ppf((1-a), alp, scale=bet)
    # test stat (m*HSIC)
    stat = 1./n * np.sum(kc.T * lc)
    return cast(
        tuple[float, float, float, float, float, float],
        (th, varHSIC, mHSIC, alp, bet, stat))


def calc_HSIC_p(x: NDArray[T], y: NDArray[T]) -> float:
    _t, _v, _m, alpha, beta, stats = test_th(x, y, bandwidth=1.)
    return cast(float, scipy.stats.gamma.cdf(stats, alpha, scale=beta))
