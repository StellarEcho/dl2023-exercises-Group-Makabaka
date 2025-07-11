"""Loss function modules."""

import numpy as np

from lib.activations import Softmax
from lib.network_base import Module


class CrossEntropyLoss(Module):
    """Compute the cross-entropy loss."""

    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute the cross entropy, mean over batch size.

        Args:
            preds: Model predictions(raw logits) with shape (batch_size, num_classes)
            labels: One-hot encoded ground truth labels with shape (batch_size, num_classes)

        Returns:
            crossentropyloss: float representing the loss summed over all classes

        Note: CrossEntropyLoss has softmax already implemented that is why we need the raw logits
        """
        assert len(preds.shape) == 2, (
            "Predictions should be of shape (batchsize, num_classes) "
            f"but are of shape {preds.shape}")
        assert len(labels.shape) == 2, (
            "Labels should be of shape (batchsize, num_classes) "
            f"but are of shape {labels.shape}")
        assert preds.shape == labels.shape, (
            "Predictions and labels should be of same shape but are "
            f"of shapes {preds.shape} and {labels.shape}")
        preds = self.softmax(preds)
        self.input_cache = preds, labels
        # START TODO #################
        # compute the loss and average it over the batch.
        # raise NotImplementedError
        # num_classes = preds.shape[1]
        # print("num_classes : ", num_classes)
        # loss = np.zeros(num_classes)
        # for i in range(num_classes):
        loss = - np.sum(labels * np.log(preds), axis=1)
        # loss = - (labels[:, 1] * np.log(preds[:, 1]) + (1 - labels[:, 1]) * np.log(1 - preds[:, 1]))
        # batch_size = preds.shape[0]
        # cEL = loss / batch_size
        cEL = np.mean(loss)
        return cEL
        # return loss
        # END TODO ##################

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Calculate the backward pass of the cross-entropy loss.

        Args:
            grad: The gradient of the following layer.

        Returns:
            The gradient of this module.
        """
        # we do not need the backward pass yet
        raise NotImplementedError


class MSELoss(Module):
    """Compute the mean squared error loss."""

    def forward(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute the mean squared error loss, mean over batch size.

        Args:
            preds: Model predictions with shape (batch_size, num_classes)
            labels: Ground truth labels with shape (batch_size, num_classes)

        Returns:
            MSE loss.
        """
        self.input_cache = preds, labels
        return np.sum(0.5 * np.linalg.norm(preds - labels, axis=-1) ** 2) / len(preds)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Calculate the backward pass of the mean squared error loss.

        Args:
            grad: The gradient of the following layer.

        Returns:
            The gradient of this module.
        """
        # we do not need the backward pass yet
        raise NotImplementedError
