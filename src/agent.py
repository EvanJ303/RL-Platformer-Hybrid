# PyTorch deep learning library
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
# Numpy for numerical operations
import numpy as np
# Custom model module
import model
# Collections for the replay memory
from collections import deque, namedtuple
# Random for sampling experiences
import random

# Named tuple for storing experiences in the replay memory
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

# Replay memory class
class ReplayMemory:
    # Initialize the replay memory with a maximum capacity
    def __init__(self, max_capacity):
        self.memory = deque([], maxlen=max_capacity)

    # Push a transition into the memory
    def push(self, *args):
        self.memory.append(Experience(*args))

    # Sample a batch of random experiences from the memory
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # Check the length of the memory
    def __len__(self):
        return len(self.memory)

# Deep Q-Network (DQN) agent class
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        # Initialize state and action dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Initialize batch size
        self.batch_size = 64
        # Initialize epsilon-greedy parameters
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon = self.epsilon_start
        self.epsilon_decay = 0.995
        # Initialize gamma, tau, and learning rate
        self.gamma = 0.99
        self.tau = 0.005
        self.lr = 0.0005
        # Initialize replay memory with a maximum capacity
        self.memory = ReplayMemory(150000)

        # Set the device to GPU if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the policy network
        self.policy_net = model.DQN(state_dim, action_dim)
        self.policy_net.to(self.device)

        # Initialize and configure the target network
        self.target_net = model.DQN(state_dim, action_dim)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Initialize the optimizer and loss function
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

    # Select an action based on the current state and epsilon-greedy policy
    def select_action(self, state):
        # Select a random action with probability epsilon
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            # Convert the state to a tensor and add a batch dimension
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            state = state.unsqueeze(0)

            # Disable gradient calculation 
            with torch.no_grad():
                # Get the Q-values from the policy network and select the action with the highest Q-value
                q_values = self.policy_net(state)
                q_values = q_values.squeeze(0)
                action = q_values.argmax().item()

        # Return the selected action
        return action
    
    # Decay the epsilon value for the epsilon-greedy policy
    def decay_epsilon(self):
        # Decay epsilon using the decay factor while ensuring it does not go below the minimum epsilon value
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    # Store an experience in the replay memory
    def store_experience(self, *args):
        # Call the push method of the replay memory to store the experience
        self.memory.push(*args)

    # Sample a random batch of experiences from the replay memory
    def sample_memory(self, batch_size):
        # Call the sample method of the replay memory to get a batch of experiences
        return self.memory.sample(batch_size)

    # Update the target network parameters using a soft update
    def update_target(self):
        # Iterate through the parameters of the target network and copy the values from the policy network
        for policy_param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
        # Set the target network back to evaluation mode
        self.target_net.eval()

    # Optimize the model using a batch of experiences from the replay memory
    def optimize_model(self):
        # If the memory has fewer experiences than the batch size, return early
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a random batch of experiences from the memory
        experiences = self.sample_memory(self.batch_size)
        batch = Experience(*zip(*experiences))

        # Check if the next state is terminal and create a mask for non-terminal states
        not_terminal_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool, device=self.device)
        # Convert the next states to a tensor, excluding terminal states
        not_terminal_next_states = torch.tensor([s for s in batch.next_state if s is not None], dtype=torch.float32, device=self.device)

        # Convert the state, action, and reward batches to tensors
        state_batch = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)

        # Calculate the Q-values of the state-action pairs using the policy network
        q_state_action = self.policy_net(state_batch).gather(1, action_batch)

        # Initialize a tensor for the next state Q-values
        q_next_state = torch.zeros(self.batch_size, device=self.device)

        # Check if there are non-terminal next states
        if not_terminal_mask.any():
            # Disable gradient calculation
            with torch.no_grad():
                # Calculate the maximum Q-values for the non-terminal next states using the target network
                q_next_state[not_terminal_mask] = self.target_net(not_terminal_next_states).max(1).values

        # Calculate the expected Q-values of the state-action pairs using the Bellman equation
        expected_q_state_action = reward_batch + (self.gamma * q_next_state)
        # Reshape the expected Q-values to match the shape of q_state_action
        expected_q_state_action = expected_q_state_action.unsqueeze(1)

        # Calculate the loss between the predicted Q-values and the expected Q-values
        loss = self.criterion(q_state_action, expected_q_state_action)

        # Zero the gradients
        self.optimizer.zero_grad()
        # Perform the backward pass to compute gradients
        loss.backward()
        # Clip the gradients to prevent exploding gradients
        utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        # Use the optimizer to update the policy network parameters
        self.optimizer.step()
        # Update the target network parameters
        self.update_target()
    
    # Set the agent to inference mode (use the final epsilon value)
    def inference_mode(self):
        # Set epsilon to the end value
        self.epsilon = self.epsilon_end

    # Set the agent to training mode (reset epsilon to the start value)
    def training_mode(self):
        # Reset epsilon to the start value
        self.epsilon = self.epsilon_start

    # Save the state dictionaries of the policy network, the target network, the optimizer, and the epsilon value to a checkpoint file
    def save(self, checkpoint_path):
        torch.save({
            'policy_net_state_dict' : self.policy_net.state_dict(),
            'target_net_state_dict' : self.target_net.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'epsilon' : self.epsilon,
        }, checkpoint_path)

    # Load the state dictionaries of the policy network, the target network, the optimizer, and the epsilon value from a checkpoint file
    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']