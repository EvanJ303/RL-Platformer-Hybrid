# PyTorch deep learning library
import torch.nn as nn
import torch.nn.functional as F

# Deep Q-Network (DQN) model class
# This class defines the architecture of the DQN used by the agent
class DQN(nn.Module):
    def __init__(self, d_state, d_action):
        # Follow the parent class constructor
        super().__init__()
        # Initialize three linear layers
        self.layer1 = nn.Linear(d_state, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, d_action)

    # Define the forward pass
    def forward(self, x):
        # Pass the input through the layers and apply ReLU activation
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        # Return the output
        return x   