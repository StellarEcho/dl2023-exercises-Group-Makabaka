"""CNN models to train"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet1(nn.Module):
    """
        The CNN model with 3 filters, kernel size 5, and padding 2
    """

    def __init__(self):
        super().__init__()
        # START TODO #############
        # initialize required parameters / layers needed to build the network
        # raise NotImplementedError
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)
        self.fc = nn.Linear(in_features=16*16*3, out_features=10)
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape [batch_size, *feature_dim] (minibatch of data)
        Returns:
            scores: Pytorch tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        # raise NotImplementedError
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # END TODO #############
        return x


class ConvNet2(nn.Module):
    """
        The CNN model with 16 filters, kernel size 5, and padding 2
    """

    def __init__(self):
        super().__init__()
        # START TODO #############
        # raise NotImplementedError
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.fc = nn.Linear(in_features=16*16*16, out_features=10)
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (batch_size, *feature_dim)
            The input to the network will be a minibatch of data

        Returns:
            scores: PyTorch Tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        # raise NotImplementedError
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # END TODO #############
        return x


class ConvNet3(nn.Module):
    """
        The CNN model with 16 filters, kernel size 3, and padding 1
    """

    def __init__(self):
        super().__init__()
        # START TODO #############
        # Define the layers need to build the network
        # raise NotImplementedError
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.fc = nn.Linear(in_features=16*16*16, out_features=10)
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (batch_size, *feature_dim)
            The input to the network will be a minibatch of data

        Returns:
            scores: PyTorch Tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        # raise NotImplementedError
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # END TODO #############
        return x


class ConvNet4(nn.Module):
    """
        The CNN model with 16 filters, kernel size 3, padding 1 and batch normalization
    """

    def __init__(self):
        super().__init__()
        # START TODO #############
        # Define the layers need to build the network
        # raise NotImplementedError
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=16)
        self.fc = nn.Linear(in_features=16*16*16, out_features=10)
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (batch_size, *feature_dim)
            The input to the network will be a minibatch of data

        Returns:
            scores: PyTorch Tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        # raise NotImplementedError
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # END TODO #############
        return x


class ConvNet5(nn.Module):
    """ Your custom CNN """

    def __init__(self):
        super().__init__()

        # START TODO #############
        # raise NotImplementedError
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.fc = nn.Linear(in_features=128*8*8, out_features=512)
        self.fc2 = nn.Linear(512, 10)
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (batch_size, *feature_dim)
            The input to the network will be a minibatch of data

        Returns:
            scores: PyTorch Tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        # raise NotImplementedError
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128*8*8)
        x = self.fc(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        # END TODO #############
        return x
