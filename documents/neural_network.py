"""Example of a simple neural network with PyTorch.

Author: Elie KADOCHE.
"""

import torch


class Net(torch.nn.Module):
    """Network class."""

    def __init__(self):
        """Init."""
        super(Net, self).__init__()
        self.A = torch.nn.Linear(1, 32)
        self.B = torch.nn.Linear(32, 32)
        self.C = torch.nn.Linear(32, 1)

    def forward(self, x):
        """Forward."""
        x = torch.relu(self.A(x))
        x = torch.relu(self.B(x))
        x = self.C(x)
        return x


if __name__ == "__main__":
    # Create model
    model = Net()

    # Create the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    # Training loop
    while True:

        # We create some random integergers. Careful here, the requires_grad is
        # very important. If we do all the computations without gradients it
        # will naturally not work
        x = torch.randint(
            0, 7, (250, 1), dtype=torch.float32, requires_grad=True)

        # We create the real y values. The objectif for the model is to give
        # the squared value of x
        y_true = torch.square(x)

        # We use our model to predict the y values
        y = model(x)

        # We compute the loss
        loss = torch.mean(torch.square(y - y_true))

        # Print loss: it should decreases!
        print("{:.2E}".format(loss.item()))

        # Reset the gradients to 0
        optimizer.zero_grad()

        # Compute the gradients of the model parameters relative to the loss
        loss.backward()

        # Update the network
        optimizer.step()
