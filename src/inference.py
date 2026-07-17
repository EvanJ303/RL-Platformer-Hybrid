# Itertools for iteration
from itertools import count

# Custom DQN agent
from agent import DQNAgent
# Custom environment
import environment

# Set the number of episodes
NUM_EPISODES = 5

def main():
    # Determine the latest checkpoint path using the text file
    with open('./data/latest_checkpoint.txt', 'r') as f:
        checkpoint_path = f.read().strip()

    # Initialize DQN agent with a state size of 6 and an action size of 3
    agent = DQNAgent(6, 3)
    # Load the checkpoint
    agent.load(checkpoint_path)
    # Set the agent to inference mode
    agent.inference_mode()

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
            
            # Update the state
            state = next_state

            # Check if the episode is over
            if done:
                # Print the episode and episode reward
                print(f'Episode {episode + 1} finished. Total reward: {episode_reward}')
                break

if __name__ == '__main__':
    main()