"""LSTM and noise removal models"""

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """The LSTM layer."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # You should define a single linear layer which takes the current input
        # and the hidden state as input, and outputs the linear transformation of all gates
        # You will chunk the output of the linear layer to four predictions
        # (candidate_cell_state, forget_gate, input_gate, output_gate) (we recommend that order)
        # during the forward propagation
        # we use hidden_size * 4 units as we can chunk/split the output later
        self.linear = nn.Linear(input_size + hidden_size, hidden_size * 4, bias=True)

    def forward(self, x: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x: The input tensor with shape (batch_size, feature_dim)
            hx: The initial hidden state, optional. Is a two-tuple consisting of
                the current hidden state with shape (batch_size, hidden_size)
                the previous cell state (C_{t-1}) with shape (batch_size, hidden_size)

        Returns:
            Tuple of:
                A torch.Tensor of the new hidden state with shape (batch_size, hidden_size)
                A torch.Tensor of the current cell state (C_{t}) with shape (batch_size, hidden_size)
        """

        if hx is None:
            hx = self._init_hidden_state(x)
        hidden_state, previous_cell_state = hx

        # START TODO #############
        # compute the gating and cell update vectors. use torch cat to merge input and hidden state
        # split the output into the four predictions (candidate_cell_state, forget_gate, input_gate, output_gate).
        # use torch chunk to split tensor
        # update the internal and hidden state
        # we intend you to not use any activation functions in this part of the code, but rather in the update function
        # Concatenate input and hidden state
        combined = torch.cat((x, hidden_state), dim=1)
        output = self.linear(combined)
        candidate_cell_state, forget_gate, input_gate, output_gate = torch.chunk(output, 4, dim=1)
        current_cell_state = self.update_internal_state(forget_gate,
                                                        previous_cell_state, input_gate, candidate_cell_state)
        new_hidden_state = self.update_hidden_state(current_cell_state, output_gate)
        # raise NotImplementedError
        # END TODO #############
        return new_hidden_state, current_cell_state

    def update_internal_state(self, forget_gate: torch.Tensor, previous_cell_state: torch.Tensor,
                              input_gate: torch.Tensor, candidate_cell_state: torch.Tensor) -> torch.Tensor:
        """
        Update the internal state based on the equation given in the slides.

        Notes:
            Here we follow the updating rules in the lecture slides, which is different from DL book.

        Args:
            forget_gate: A torch.Tensor which becomes forget gate (f_t) after passing through activation
                function with shape (batch_size, hidden_size)
            previous_cell_state: A torch.Tensor of the previous cell state (C_{t-1}) with
                shape (batch_size, hidden_size)
            input_gate: A torch.Tensor which becomes external input gate tensor (i_t)
                after passing through activation function with shape (batch_size, hidden_size)
            candidate_cell_state: A torch.Tensor which becomes candidate cell state (\tilde{C}_t) after
                passing through activation function with shape (batch_size, hidden_size)

        Returns:
            A torch.Tensor of the current cell state with shape (batch_size, hidden_size)
        """
        # START TODO #############
        # calculate the new internal state, applying the activation functions to the respective tensors
        candidate_cell_state = torch.tanh(candidate_cell_state)
        forget_gate = torch.sigmoid(forget_gate)
        input_gate = torch.sigmoid(input_gate)
        # Update cell state
        current_cell_state = (forget_gate * previous_cell_state) + (input_gate * candidate_cell_state)
        # raise NotImplementedError
        # END TODO #############
        return current_cell_state

    def update_hidden_state(self, current_cell_state: torch.Tensor,
                            output_gate: torch.Tensor) -> torch.Tensor:
        """
        Update the hidden state based on the equation given in the slides.

        Args:
            current_cell_state: A torch.Tensor of the current cell state (C_{t})
                with shape (batch_size, hidden_size)
            output_gate: A torch.Tensor which becomes output gate tensor (o_t)
                after passing through activation function with shape (batch_size, hidden_size)

        Returns:
            A torch.Tensor of the new hidden state (h_t) with shape (batch_size, hidden_size)
        """
        # START TODO #############
        # calculate the new hidden state, applying the activation functions to the respective tensors
        output_gate = torch.sigmoid(output_gate)
        new_hidden_state = output_gate * torch.tanh(current_cell_state)
        # raise NotImplementedError
        # END TODO #############
        return new_hidden_state

    def _init_hidden_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the hidden state with zeros.

        Returns:
            A two-tuple (initial_hidden_state with shape (batch_size, hidden_size),
                initial_cell_state with shape (batch_size, hidden_size)).
        """

        # # START TODO #############
        initial_hidden_state = torch.zeros(x.shape[0], self.hidden_size)
        initial_cell_state = torch.zeros(x.shape[0], self.hidden_size)

        # raise NotImplementedError
        # END TODO #############
        return initial_hidden_state, initial_cell_state


class LSTM(nn.Module):
    """
    Convenience class that automatically iterates over the sequence.

    Args:
        input_size: Input dimension.
        hidden_size: Hidden dimension.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm_cell = LSTMCell(input_size, hidden_size)

    def forward(self, x: Union[np.ndarray, torch.Tensor], hx=None) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over the sequence and apply the LSTM cell.

        Args:
            x: The input tensor with shape (batch, seq_len, input_size)
            hx: The initial hidden state, optional. Is a two-tuple consisting of
                the current hidden state and the internal cell state. Both have
                shape (batch_size, hidden_size). If None, set to zero.

        Returns:
            Tuple of:
                output_stacked_hidden, the stacked output of all LSTMCells with shape (batch, seq_len, hidden_size)
                    (excluding the cell state!)
                Tuple of:
                    last_hidden_state with shape (batch_size, hidden_size)
                    last_new_internal_state with shape (batch_size, hidden_size)
        """

        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x, requires_grad=True)

        # START TODO #############
        seq_len = x.shape[1]
        # if hx is None:
        #     hx = self.lstm_cell._init_hidden_state(x)
        output_stacked_hidden = []
        for i in range(seq_len):
            hx = self.lstm_cell(x[:, i, :], hx)
            output_stacked_hidden.append(hx[0])
        return torch.stack(output_stacked_hidden, dim=1), hx
        # raise NotImplementedError
        # END TODO #############


class NoiseRemovalModel(nn.Module):
    """
    Model which uses LSTMs to remove noise from a noisy signal.

    Args:
        hidden_size: The number of units of the LSTM hidden state size.
        shift: The number of steps the RNN is run before its output is considered ("many-to-many shifted to the right").
    """

    def __init__(self, hidden_size: int, shift: int = 10):
        super().__init__()
        self.shift = shift
        # START TODO #############
        # Create the 2 LSTM and 1 Linear module as described in the assignment.
        self.lstm1 = LSTM(1, hidden_size)
        self.lstm2 = LSTM(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

        # raise NotImplementedError
        # END TODO #############

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of noise removal.

        This function
        1) pads the input sequence with self.shift zeros at the end,
        2) perform forward passing of the first LSTM
        3) cuts the first self.shift outputs
        4) perform forward passing of the second LSTM
        5) applies Linear layer.

        Args:
            x: The input tensor with shape (batch_size, sequence length, 1)

        Returns:
            A torch.Tensor of shape (batch_size, sequence length, 1)
        """

        # Pad input sequence x at the end (shifted many-to-many model).
        # This allows the model to see a few numbers before it has to guess
        # the noiseless output.

        # START TODO #############
        batch_size, seq_len, input_size = x.shape

        x_padded = torch.cat([x, torch.zeros(batch_size, self.shift, input_size)], dim=1)

        out_lstm1, _ = self.lstm1(x_padded)

        out_lstm1_trimmed = out_lstm1[:, self.shift:, :]
        out_lstm2, _ = self.lstm2(out_lstm1_trimmed)

        output = self.linear(out_lstm2)

        return output
        # raise NotImplementedError
        # END TODO #############
