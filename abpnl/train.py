from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import os
import sys
from typing import Any, TypeVar, cast
from collections.abc import Sequence, Callable
import random as py_random
from concurrent.futures import Future, ProcessPoolExecutor

import numpy as np
from numpy.typing import NDArray

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.optim import Adam

from abpnl import AbPNLModel, NtoOneDataset
import stats

# T = TypeVar('T', bound=np.generic, covariant=True)
T = TypeVar('T', bound=np.floating[Any])
SEP = os.path.sep


def pp(*args: Any, **kwargs: Any) -> None:
    print(*args, **kwargs)
    sys.stdout.flush()


class LogItems(defaultdict[int, dict[str, Any]]):
    def seq(self, key: str) -> list[tuple[int, Any]]:
        return [
            (_, self[_][key]) for _ in sorted(self.keys()) if key in self[_]
            ]


def to_csv(
        log: dict[int, dict[str, Any]], sep: str = ",", lastln: bool = True
        ) -> str:
    """
    Convert logs to csv-formatted string.

    Parameters
    ----
    log : dict
        key=iteration, value=dict(str, numeric)
    sep : str
        delimiter string
    """
    columns_ = set()
    for _, l in log.items():
        for c in l.keys():
            if c not in columns_:
                columns_.add(c)
    columns = sorted(list(columns_))

    s = [sep.join(["idx", ] + columns), ]
    for i, l in log.items():
        s.append(
            sep.join([str(i), ] + [str(l.get(_, "")) for _ in columns])
        )
    return "\n".join(s) + ("\n" if lastln else "")


class AbPNLTrainer(object):
    def __init__(self, params: dict[str, Any]):
        super().__init__()
        self._params = params
        self._causal_order: list[int] = []
        self._adjacency_matrix: NDArray[Any]

    def _check_params(self) -> None:
        keys = {
            "logdir": "str. A path of log directory.",
            "max_workers": "int. The number of parallel execution.",
            "n_trials": "int, The number of trials for one model learning.",
            "pruning_threshold":
                "float. Threshold of p-value used for pruning.",
            "n_epoch": "int. The number of learning epoch.",
            "n_units": "int. The number of units in a layer of MLPs.",
            "n_layers": "int. The number of layers in MPLs.",
            "activation": "Callable. The activation function.",
            "loss_balance":
                "float. Relative weight of loss_ind against loss_inv.",
            "batchsize": "int. Batchsize.",
            "dropout": "bool. Use dropout.",
            "optimizer": "Callable. Optimizer.",
            "learning_rate": "float. Learning rate for the optimizer.",
            "interval_test": "int. The interval of epochs at which to test",
            "interval_save":
                "int. The interval of epochs at which to save the model."
        }
        for k, v in keys.items():
            if k not in self._params:
                raise ValueError(f"`{k}` ({v}) not in params.")

    def doit(
            self,
            x_train: Sequence[Sequence[T]] | NDArray[T],
            x_test: Sequence[Sequence[T]] | NDArray[T]
            ) -> None:
        """

        Parameters
        ----------
        x_train : Sequence[Sequence[T]]
            Training samples with shape=(#samples, #variables).
        x_test : Sequence[Sequence[T]]
            Test samples with shape=(#samples, #variables).
        """
        self._x_train = cast(Sequence[Sequence[T]], x_train)
        self._x_test = cast(Sequence[Sequence[T]], x_test)
        self._variables = list(range(len(x_train[0])))

        os.makedirs(self._params["logdir"], exist_ok=True)
        while len(self._variables) > 1:
            self._find_sink()
        top = self._variables[0]
        self._causal_order.append(top)
        self._prune()

    def _find_sink(self) -> None:
        logs = self._train_all_ntoone_parallel()
        sink = self._select_var(logs, "loss/test",
                                "loss_ind/test", np.median)[0]

        self._variables = [_ for _ in self._variables if _ != sink]
        self._causal_order.append(sink)
        pp("causal order:", self._causal_order)

        # logging
        pstfx = "_".join(f"{_}" for _ in self._variables)
        # selected models
        logfn = self._params["logdir"] + f"{SEP}models_{pstfx}.csv"
        csv = to_csv({_i: _d for _i, _d in enumerate(self._log_models)})
        with open(logfn, "w") as f:
            f.write(csv)
        # stats
        logfn = self._params["logdir"] + f"{SEP}stats_{pstfx}.csv"
        csv = "\n".join(
            ",".join(f"{_}" for _ in _l) for _l in self._log_stats
        ) + "\n"
        with open(logfn, "w") as f:
            f.write(csv)

    def _train_all_ntoone_parallel(self) -> dict[int, list[LogItems]]:
        """ Train all the possible n-to-one pnl models. """
        futures: list[tuple[int, int, Future[LogItems]]] = []
        with ProcessPoolExecutor(
                max_workers=self._params["max_workers"]) as executor:
            for effect in self._variables:
                causes = [_ for _ in self._variables if _ != effect]
                ds_train = NtoOneDataset(self._x_train, causes, effect)
                ds_test = NtoOneDataset(self._x_test, causes, effect)
                parent_logdir = "cause" + "_".join(str(_) for _ in causes) + \
                                f"effect{effect}"
                for t in range(self._params["n_trials"]):
                    logdir = f"{parent_logdir}{SEP}trial{t}"
                    self._params["l_logdir"] = \
                        self._params["logdir"] + SEP + logdir
                    self._params["l_seed"] = np.random.randint(1000000)
                    params = deepcopy(self._params)
                    if not os.path.exists(params["l_logdir"]):
                        os.makedirs(params["l_logdir"])

                    future = executor.submit(train, ds_train, ds_test, params)
                    futures.append((effect, t, future))
            results = [(_[0], _[1], _[2].result()) for _ in futures]  # Wait.

        all_logs: dict[int, list[LogItems | None]] = {
            _: [None]*self._params["n_trials"] for _ in self._variables
        }
        for effect, t, log in results:
            all_logs[effect][t] = log
        return cast(dict[int, list[LogItems]], all_logs)

    def _select_var(
            self,
            logs: dict[int, list[LogItems]],
            key_model: str = "loss/test",
            key_stats: str = "loss_ind/test",
            f_stats: Callable[[list[float]], float] = np.median
            ) -> tuple[int, float]:
        """Select an index with the minimum criterion calculated over trials.

        Parameters
        ----------
        logs : dict[int, list[LogItems]]
            All LogItems.
        key_model : str, optional
            Key to select a model in one trial, by default "loss/test"
        key_stats : str, optional
            Key to calculate stats over trials, by default "loss_ind/test"
        f_stats : Callable[[list[Any]], Any], optional
            Stats function over trials, by default np.median

        Returns
        -------
        int
            Index that achieved the minimum stats value.
        Any
            Its value.
        """
        models: list[dict[str, Any]] = []
        measures: list[tuple[int, float]] = []
        for e, log_e in logs.items():
            measures_t = []
            for t, log in enumerate(log_e):
                i = self._select_model(log, key_model)
                record = log[i]  # Log record of the selected model.

                # AbPNL sometimes fails to reconstruct the effect, y'.
                # Remove such trials from calculating stats of the model.
                # In such trials, y' becomes a constant, thus MSE~var(y).
                rec_ok = \
                    record["loss_inv/test"] < .5 * record["y_stats/variance"]
                if rec_ok:
                    measures_t.append(record[key_stats])
                model = {
                    "__condition": e, "_trial": t, "_reconstruct": rec_ok
                }
                model.update(record)
                models.append(model)
            measures.append((
                e, f_stats(measures_t) if len(measures_t) > 0 else float("inf")
            ))
        self._log_models = models
        self._log_stats = measures
        k = np.argmin([_[1] for _ in measures])
        return measures[k]

    def _select_model(self, log: LogItems, key: str = "loss/test") -> int:
        return min(log.seq(key), key=lambda _: cast(float, _[1]))[0]

    def _prune(self) -> None:
        p = len(self._causal_order)
        parents = {
            self._causal_order[_]: self._causal_order[_+1:] for _ in range(p)
        }
        pruning_finished = {_: False for _ in self._causal_order}
        pruning_finished[self._causal_order[-1]] = True

        while not np.all(list(pruning_finished.values())):
            with ProcessPoolExecutor(
                    max_workers=self._params["max_workers"]) as executor:
                futures: list[tuple[int, int, list[int],
                                    int, Future[LogItems]]] = []
                for i, parents_i in parents.items():
                    fin_i = pruning_finished[i]
                    if not fin_i:
                        for exclude_parent in parents_i:
                            parents_i_new = [
                                _ for _ in parents_i if _ != exclude_parent]

                            if len(parents_i_new) > 0:
                                logdir_prfx = \
                                    f"pruning{SEP}cause{i}effect" + \
                                    "_".join(str(_) for _ in parents_i_new) + \
                                    f"remove{exclude_parent}"
                                for t in range(self._params["n_trials"]):
                                    logdir = self._params["logdir"] + \
                                        f"{SEP}{logdir_prfx}{SEP}trial{t}"
                                    seed = np.random.randint(1000000)

                                    self._params["l_logdir"] = logdir
                                    self._params["l_seed"] = seed
                                    params = deepcopy(self._params)
                                    if not os.path.exists(logdir):
                                        os.makedirs(logdir)

                                    future = executor.submit(
                                        evaluate_prune,
                                        self._x_train, self._x_test,
                                        i, parents_i_new, exclude_parent,
                                        params, t
                                    )
                                    futures.append((i, exclude_parent,
                                                    parents_i_new, t, future))
                            else:
                                logdir = \
                                    self._params["logdir"] + \
                                    f"{SEP}pruning{SEP}" + \
                                    f"cause{i}remove{exclude_parent}"
                                self._params["l_logdir"] = logdir
                                params = deepcopy(self._params)
                                t = 0
                                future = executor.submit(
                                    evaluate_prune,
                                    self._x_train, self._x_test,
                                    i, parents_i_new, exclude_parent,
                                    params, t
                                )
                                futures.append((i, exclude_parent,
                                                parents_i_new, t, future))
                results = {
                    (_[0], _[1], _[3]): (_[2], _[4].result()) for _ in futures
                    }  # {(child, excluded_parent, trial):(parents, results)}

            prune_logs: dict[int, dict[int, list[LogItems | None]]]
            prune_logs = defaultdict(dict)
            for (i, j, t), (pa, log) in results.items():
                if j not in prune_logs[i]:
                    if len(pa) > 0:
                        prune_logs[i][j] = [None, ] * self._params["n_trials"]
                    else:
                        prune_logs[i][j] = [None, ]
                prune_logs[i][j][t] = log
            for i,  prune_log in prune_logs.items():
                j, p_val = self._select_var(
                    cast(dict[int, list[LogItems]], prune_log),
                    "loss/test", "comp/hsic/p", np.min)

                # logging start
                pstfx = "cause" + "_".join(f"{_}" for _ in parents[i]) + \
                        f"effect{i}"
                # selected models
                logfn = self._params["logdir"] + \
                    f"{SEP}pruning{SEP}models_{pstfx}.csv"
                csv = to_csv({
                    _i: _d for _i, _d in enumerate(self._log_models)
                })
                with open(logfn, "w") as f:
                    f.write(csv)
                # stats
                csv = "\n".join(
                    ",".join(f"{_}" for _ in _l) for _l in self._log_stats
                    ) + "\n"
                logfn = self._params["logdir"] + \
                    f"{SEP}pruning{SEP}measures_{pstfx}.csv"
                with open(logfn, "w") as f:
                    f.write(csv)
                # logging end

                threshold = self._params["pruning_threshold"]
                if p_val < threshold:
                    parents[i] = [_ for _ in parents[i] if _ != j]
                    if len(parents[i]) == 0:
                        pruning_finished[i] = True
                else:
                    pruning_finished[i] = True
        adjacency_matrix = np.zeros((p, p), dtype=int)
        for i, parents_i in parents.items():
            for j in parents_i:
                adjacency_matrix[i, j] = 1
        self._adjacency_matrix = adjacency_matrix

    @property
    def causal_order(self) -> list[int]:
        return self._causal_order

    @property
    def adjacency_matrix(self) -> NDArray[T]:
        return self._adjacency_matrix

    default_params = {
        "logdir": "abpnl_results",
        "max_workers": 1,
        "n_trials": 9,
        "pruning_threshold": 0.95,
        "n_epoch": 100,
        "n_units": 5,
        "n_layers": 3,
        "activation": F.leaky_relu,
        "loss_balance": 0.5,
        "batchsize": 64,
        "dropout": True,
        "optimizer": Adam,
        "learning_rate": 1e-3,
        "interval_test": 1,
        "interval_save": -1,
    }


def train(
        x_train: NtoOneDataset[T], x_test: NtoOneDataset[T],
        params: dict[str, Any],
        noise_stats_samples: NDArray[T] | None = None
        ) -> LogItems:
    seed = params["l_seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    py_random.seed(seed)

    pp(params["l_logdir"])

    train_loader = DataLoader(x_train, batch_size=params["batchsize"],
                              shuffle=True)
    test_loader = DataLoader(x_test, batch_size=len(x_test))

    model = AbPNLModel(
        nx=len(x_train.causes), ny=1,
        nz=params["n_units"], nl=params["n_layers"],
        actf=params["activation"], a=params["loss_balance"],
    )
    opt = cast(
        Optimizer,
        params["optimizer"](model.parameters(), params["learning_rate"])
        )
    log_items = LogItems(dict)

    iteration = -1
    for epoch in range(params["n_epoch"]):
        for x, y in train_loader:
            iteration += 1

            opt.zero_grad()
            loss = model(x, y)
            loss.backward()
            opt.step()

            log_items[iteration].update({
                "epoch": epoch, "iteration": iteration,
                "loss/train": model.loss,
                "loss_ind/train": model.loss_ind,
                "loss_inv/train": model.loss_inv,
            })

        if (epoch+1) % params["interval_test"] == 0:
            x, y = test_loader.__iter__().next()  # Get all the test samples.
            model.train(False)
            with torch.no_grad():
                loss = model(x, y)
            model.train(True)

            log_items[iteration].update({
                "loss/test": model.loss,
                "loss_ind/test": model.loss_ind,
                "loss_inv/test": model.loss_inv,
            })

            # Will be used for checking reconstruction failure.
            arr_y = y.detach().numpy()
            log_items[iteration].update({
                "y_stats/mean": arr_y.mean(),
                "y_stats/variance": arr_y.var(),
            })

            if noise_stats_samples is not None:
                z = stats.standardize(noise_stats_samples)
                e = stats.standardize(model.e)
                p_val = stats.calc_HSIC_p(z, e)
                log_items[iteration].update({
                    "comp/hsic/p": p_val
                })

        if (params["interval_save"] > 0) and \
           ((epoch+1) % params["interval_save"] == 0):
            torch.save(
                model.state_dict(),
                params["l_logdir"]
                + f"{SEP}model_iter{iteration}"
                )

    torch.save(
        model.state_dict(),
        params["l_logdir"] + f"{SEP}model_final")

    csv = to_csv(log_items)
    with open(params["l_logdir"] + f"{SEP}log.csv", "w") as f:
        f.write(csv)

    return log_items


def evaluate_prune(
        x_train: Sequence[Sequence[T]], x_test: Sequence[Sequence[T]],
        child: int, parents: list[int],
        excluded_parent: int, params: dict[str, Any], t: int
        ) -> LogItems:
    if len(parents) > 0:
        ds_train = NtoOneDataset(x_train, parents, child)
        ds_test = NtoOneDataset(x_test, parents, child)
        z = np.array(x_test)[:, excluded_parent]

        logs = train(ds_train, ds_test, params, z)
    else:
        x = np.array(x_test)[:, excluded_parent]
        y = np.array(x_test)[:, child]

        p_val = stats.calc_HSIC_p(stats.standardize(x), stats.standardize(y))
        logs = LogItems(dict)
        logs[0].update({
            "comp/hsic/p": p_val
        })
        # Dummy
        logs[0].update({
            "loss/test": 0.0,
            "loss_inv/test": 0.0,
            "y_stats/variance": 1.0,
        })

        os.makedirs(params["l_logdir"], exist_ok=True)
        csv = to_csv(logs)
        with open(params["l_logdir"] + f"{SEP}log.csv", "w") as f:
            f.write(csv)

    return logs
