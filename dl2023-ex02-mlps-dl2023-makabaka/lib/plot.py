"""ReLU plot."""

import matplotlib.pyplot as plt
import numpy as np

from lib.activations import ReLU


def plot_relu() -> None:
    """Plot the ReLU function in the range (-4, 4).

    Returns:
        None
    """
    # START TODO #################
    # Create input data, run through ReLU and plot.
    x = np.linspace(-4, +4, 101)
    func_ReLU = ReLU()
    y = func_ReLU(x)
    plt.plot(x, y)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.grid(True)
    plt.show()
    # raise NotImplementedError
    # END TODO###################
