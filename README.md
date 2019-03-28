# gaussian-process

An implementation of Gaussian processes in Python. A model for supervised learning, GP, as well as a model for unsupervised learning, GPLVM, are provided. Multiple kernels are implemented, along with gradients to optimise hyperparameters.

## Requirements

Running the code requires Python 3.6+ and the following packages: NumPy, Scipy, Scikit-learn and matplotlib. To install them with pip, run

```
$ pip3 install -e .
```

This will install the required packages, and this project in editable mode, making the examples runnable from terminal.

## Examples

Two examples are included, `plot_gp.py` and `plot_gplvm.py`. These can be run from terminal, and can be modified to try some other data and/or kernel.
