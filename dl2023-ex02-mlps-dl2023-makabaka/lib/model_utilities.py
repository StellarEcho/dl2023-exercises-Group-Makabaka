"""Helper functions for working with models."""

import numpy as np

from lib.network import Sequential


def extract_hidden(full_model: Sequential, x: np.ndarray) -> np.ndarray:
    """Function to extract the hidden representation from a 2-layer MLP.
    Compute `h` (hidden representation) after propagating the `x` (input)
    through the first Linear layer and the activation function ReLU
    of `full_model`.

    Args:
        full_model: The 2-layer MLP used as a classifier.
        x: Input examples with shape (nr_examples, nr_features).

    Returns:
        h: Hidden representation of inputs with shape (nr_examples, nr_features).
    """
    # START TODO #################
    # Change the weights of the first Linear layer according to the exercise sheet
    # Extract the hidden features from the 2-layer MLP and compute the hidden representation after propagating
    # the input through the first Linear layer and the activation function.
    # raise NotImplementedError
    W1 = np.array([[1, 1], [1, 1]])
    b1 = np.array([0, -1])
    full_model.parameters()[0].data = W1
    full_model.parameters()[1].data = b1
    h = full_model.modules[0](x)
    h = full_model.modules[1](h)
    return h
    # END TODO ##################
