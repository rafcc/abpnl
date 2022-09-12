from __future__ import annotations

from typing import TypeAlias, Any
import numpy as np
from numpy.typing import NDArray

from train import AbPNLTrainer
from synthproblems import PostNonlinearNToOne, generate_samples

T: TypeAlias = NDArray[np.floating[Any]]


def d_edit(a: T, b: T) -> Any:
    return np.logical_xor((a != 0), (b != 0)).sum()


def n_rev(a: T, b: T) -> Any:
    return ((a != 0) * (b != 0).T).sum()


def sample1() -> None:
    """Linear problem"""
    n = 2000
    d = 4
    rng = np.random.RandomState(seed=123)

    a = np.tril((rng.random((d, d)) > 0.3) * rng.random((d, d)), k=-1)
    e = rng.uniform(size=(d, n))
    b = c = np.eye(d)
    for i in range(d - 1):
        c += (b := b@a)
    x = c @ e
    x = x.T.astype(np.float32)
    k = int(0.5*n)
    x_train = x[:k]
    x_test = x[k:]

    params = AbPNLTrainer.default_params
    params["logdir"] = "results/eg1"
    params["max_workers"] = 15

    abpnl = AbPNLTrainer(params)
    abpnl.doit(x_train, x_test)

    co = abpnl.causal_order
    am = abpnl.adjacency_matrix

    print(co)
    print(am)
    print((a != 0.) + 0)
    print("d_edit:", d_edit(a, am), "n_rev:", n_rev(a, am))


def sample2() -> None:
    """Random nonlinear problems"""
    logdir = "results/eg2"
    params = AbPNLTrainer.default_params
    params["max_workers"] = 15
    params["n_epoch"] = 200
    results = []

    x, problem = generate_samples(
        1, 4, 2000, 0, PostNonlinearNToOne,
        tgt_snratio=1., tgt_mean=.5, tgt_var=.01
    )
    x_train = x[:1000]
    x_test = x[1000:]
    l_logdir = f"{logdir}"
    params["logdir"] = l_logdir

    abpnl = AbPNLTrainer(params)
    abpnl.doit(x_train, x_test)

    gt = problem.adj_mat
    co = abpnl.causal_order
    am = abpnl.adjacency_matrix

    np.savetxt(f"{l_logdir}/gt.csv", gt, delimiter=",", fmt='%d')
    np.savetxt(f"{l_logdir}/am.csv", am, delimiter=",", fmt='%d')
    np.savetxt(f"{l_logdir}/co.csv", np.array(co), delimiter=",", fmt='%d')

    m1 = d_edit(gt, am)
    m2 = n_rev(gt, am)
    print(gt)
    print((am != 0) + 0)
    print(co)
    print("d_edit:", m1, "n_rev:", m2)
    results.append((m1, m2))
    np.savetxt(f"{logdir}/m.csv", np.array(results), delimiter=",")
    print(results)


if __name__ == "__main__":
    sample1()
    # sample2()
