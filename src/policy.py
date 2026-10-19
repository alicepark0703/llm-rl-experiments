import torch.nn as nn


class PolicyNetwork(nn.Module):
    """Choose between direct answering and calculator use."""

    def __init__(self, hidden_size=32):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, state):
        """
        Return action logits.

        Input:
            Tensor with shape (2,) or (batch_size, 2).

        Output:
            Logits for action 0 and action 1.
        """
        return self.network(state)