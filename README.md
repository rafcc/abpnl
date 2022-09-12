# Autoencoder-based causal discovery based on multivariate post-nonlinear model

This is an official implementation of the multivariate nonlinear causal discovery method using post-nonlinear causal model in the following papers.
- [A Multivariate Causal Discovery based on Post-Nonlinear Model, CLeaR 2022](https://proceedings.mlr.press/v177/uemura22a.html)
- [Estimation Of Post-Nonlinear Causal Models Using Autoencoding Structure, ICASSP 2020](https://ieeexplore.ieee.org/document/9053468)

## Requirements
- python 3.10 (may work on >=3.8 but tested only on 3.10)
  - numpy
  - scipy
  - torch

## Example
```
./sample.sh
```
See `example.py` for more details.

Note: If you use parallelization with `max_workers` parameter, it is recommended to disable Numpy's multithreading by, for example, `export OMP_NUM_THREADS=1`.