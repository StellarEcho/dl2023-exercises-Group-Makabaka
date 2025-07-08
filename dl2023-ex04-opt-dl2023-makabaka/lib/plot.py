"""Plotting functions."""

import matplotlib.pyplot as plt
import numpy as np

from lib.lr_schedulers import PiecewiseConstantLR, CosineAnnealingLR
from lib.optimizers import Adam, SGD
from lib.network_base import Parameter
from lib.utilities import load_result


def plot_learning_curves() -> None:
    """Plot the performance of SGD, SGD with momentum, and Adam optimizers.

    Note:
        This function requires the saved results of compare_optimizers() above, so make
        sure you run compare_optimizers() first.
    """
    optim_results = load_result('optimizers_comparison')
    # START TODO ################
    # train result are tuple(train_costs, train_accuracies, eval_costs,
    # eval_accuracies). You can access the iterable via
    # optim_results.items()
    # raise NotImplementedError
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    for item in optim_results.items():
        e, m = item
        plt.plot(range(len(m[0])), m[0], label=f'Loss_{e}')
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    for item in optim_results.items():
        e, m = item
        plt.plot(range(len(m[0])), m[1], label=f'Accuracy_{e}')
    plt.tight_layout()
    plt.show()
    # END TODO ###################


def plot_lr_schedules() -> None:
    """Plot the learning rate schedules of piecewise and cosine schedulers.

    """
    num_epochs = 80
    base_lr = 0.1

    piecewise_scheduler = PiecewiseConstantLR(Adam([], lr=base_lr), [10, 20, 40, 50], [0.1, 0.05, 0.01, 0.001])
    cosine_scheduler = CosineAnnealingLR(Adam([], lr=base_lr), num_epochs)

    # START TODO ################
    # plot piecewise lr and cosine lr
    # raise NotImplementedError
    plt.figure()
    names = ["piecewise", "cosine"]
    schedulers = [piecewise_scheduler, cosine_scheduler]

    for s, sn in zip(schedulers, names):
        lr = []
        for epoch in range(num_epochs):
            s.step()
            lr.append(s.optimizer.lr)
        plt.plot(range(num_epochs), lr, label=sn)
    plt.title('Learning Rate Schedules')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # END TODO ################
