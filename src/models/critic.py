"""Critic (value) deep neural network.

Author: Elie KADOCHE.
"""

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

# ---> TODO: based on the actor network, build a critic network


class CriticModel(nn.Module):
    """Deep neural network."""

    # By default, use CPU
    DEVICE = torch.device("cpu")

    def __init__(self):
        """Initialize model."""
        super(CriticModel, self).__init__()
        # Input size
        input_size = 0

        # Build layer objects

    def _preprocessor(self, state):
        """Preprocessor function.

        Args:
            state (numpy.array): environment state.

        Returns:
            x (torch.tensor): preprocessed state.
        """
        # Add batch dimension
        x = np.expand_dims(state, 0)

        # Transform to torch.tensor
        x = torch.from_numpy(x).float().to(self.DEVICE)

        return x

    def forward(self, x):
        """Forward pass.

        Args:
            x (numpy.array): environment state.

        Returns:
            value (torch.tensor): value.
        """
        # Preprocessor
        x = self._preprocessor(x)

        # Critic
        critic = x

        return critic
