# Neural networks used by the learned agents.

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.norm1   = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.norm2   = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # The skip connection lets the deeper policy-value network stay stable
        # without every layer having to relearn the whole representation.
        residual = x
        x = F.relu(self.norm1(self.linear1(x)))
        x = self.norm2(self.linear2(x))
        return F.relu(x + residual)

class PolicyValueNetwork(nn.Module):
    # AlphaZero uses one shared trunk and two heads: one head scores moves and
    # the other estimates who is winning from this state.
    HIDDEN_SIZE = 256
    NUM_BLOCKS  = 4

    def __init__(self, state_size, action_size, hidden_size=None, num_blocks=None):
        super().__init__()
        self.hidden_size = int(hidden_size or self.HIDDEN_SIZE)
        self.num_blocks = int(num_blocks or self.NUM_BLOCKS)
        self.input_projection = nn.Linear(state_size, self.hidden_size)
        self.input_norm       = nn.LayerNorm(self.hidden_size)
        self.residual_blocks  = nn.ModuleList(
            [ResidualBlock(self.hidden_size) for _ in range(self.num_blocks)]
        )
        self.policy_head = nn.Linear(self.hidden_size, action_size)
        self.value_head  = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.input_norm(self.input_projection(x)))
        for block in self.residual_blocks:
            x = block(x)
        policy_logits = self.policy_head(x)
        value         = torch.tanh(self.value_head(x))
        return policy_logits, value

class AfterstateValueNetwork(nn.Module):
    # The afterstate agent only needs one value estimate for a candidate move,
    # so its network stays smaller and simpler than AlphaZero's.
    def __init__(self, state_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
