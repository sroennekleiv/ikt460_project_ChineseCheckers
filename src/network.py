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

    def __init__(self, state_size, action_size):
        super().__init__()
        self.input_projection = nn.Linear(state_size, self.HIDDEN_SIZE)
        self.input_norm       = nn.LayerNorm(self.HIDDEN_SIZE)
        self.residual_blocks  = nn.ModuleList(
            [ResidualBlock(self.HIDDEN_SIZE) for _ in range(self.NUM_BLOCKS)]
        )
        self.policy_head = nn.Linear(self.HIDDEN_SIZE, action_size)
        self.value_head  = nn.Linear(self.HIDDEN_SIZE, 1)

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
