# Itertools for iteration
from itertools import count
# Datetime for timestamp calculation
from datetime import datetime
# Custom DQN agent
from agent import DQNAgent
# Custom environment
import environment

# Initialize DQN agent with a state size of 6 and an action size of 3
agent = DQNAgent(6, 3)

# Set number of episodes
NUM_EPISODES = 1000
# Calculate the timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Repeat for each episode
for episode in range(NUM_EPISODES):
    # Reset the environment and receive the state
    state = environment.reset()
    # Set the episode reward to 0
    episode_reward = 0.0

    # Repeat for each step
    for step in count():
        # Select the action using the state
        action = agent.select_action(state)
        # Input the action into the environment and receive the next state, reward, and done
        next_state, reward, done = environment.step(action)

        # Update the episode reward with the reward
        episode_reward += reward

        # If the state is terminal set the next state to None
        if done:
            next_state = None

        # Store the experience
        agent.store_experience(state, action, next_state, reward)
        
        # Update the state
        state = next_state

        # Optimize the DQN model
        agent.optimize_model()

        # Check if the episode is over
        if done:
            # Print the episode and episode reward
            print(f'Episode {episode + 1} finished. Total reward: {episode_reward}.')
            # Decay the agent's epsilon value
            agent.decay_epsilon()
            break

# Create a path for the checkpoint using the timestamp
checkpoint_path = f'./data/models/dqn_agent_{timestamp}.pth'

# Save the agent in the checkpoint path
agent.save(checkpoint_path)

# Write the checkpoint path to a text file
with open('./data/latest_checkpoint.txt', 'w') as f:
    f.write(checkpoint_path)