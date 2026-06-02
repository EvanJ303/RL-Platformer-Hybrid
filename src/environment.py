# Numpy for numerical operations
import numpy as np
# Pygame for rendering and game loop
import pygame
# System for exiting the program
import sys

# Initialize screen dimensions
WIDTH = 1200
HEIGHT = 900
# Initialize Pygame and the window
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Platformer')

# Initialize the clock and set the frames per second
clock = pygame.time.Clock()
FPS = 160

# Define colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Initialize the agent's rectangle
agent = pygame.Rect(588, 300, 25, 25)
# Initialize the agent's speed and velocity
agent_speed = 5
agent_vel_y = 0
# Define gravity and jump power
GRAVITY = 0.5
JUMP_POWER = -20
# Set the on-ground state to 0 (not on ground)
on_ground = 0

# Default state of the agent and objective
# (agent_x, agent_y, agent_vel_y, objective_x, objective_y, on_ground)
DEFAULT_STATE = (588, 300, 0, 570, 305, 0)

# Define the maximum number of steps in an episode and set the step count to 0
MAX_STEPS = 3000
step_count = 0

# Initialize the ground
ground = pygame.Rect(0, 550, WIDTH, 50)

# Initialize the platforms
platforms = [
    pygame.Rect(250, 450, 200, 20),
    pygame.Rect(500, 350, 200, 20),
    pygame.Rect(750, 450, 200, 20)
]

# Initialize the objectives
objectives = [
    pygame.Rect(320, 405, 60, 60),
    pygame.Rect(570, 305, 60, 60),
    pygame.Rect(820, 405, 60, 60)
]

# Set the initial objective index
objective_index = 1

# Step function to handle the agent's actions and give feedback
def step(agent_input):
    # Access global variables
    global agent_vel_y, on_ground, objective_index, step_count

    # Check if the maximum number of steps has been reached
    done = False
    step_count += 1
    if step_count >= MAX_STEPS:
        done = True

    # Check if the game window is closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    # Calculate the distance to the objective
    prev_dist = np.sqrt((agent.x - objectives[objective_index].x) ** 2 + (agent.y - objectives[objective_index].y) ** 2)

    # Update the agent's x position based on input
    if agent_input == 0:
        agent.x -= agent_speed
    if agent_input == 1:
        agent.x += agent_speed

    # Ensure the agent stays within the screen bounds
    agent.x = max(0, min(WIDTH - agent.width, agent.x))
    agent.y = max(0, min(HEIGHT - agent.height, agent.y))

    # Handle jumping
    if agent_input == 2 and on_ground:
        agent_vel_y = JUMP_POWER
        on_ground = 0

    # Calculate the gravitational effect on the agent
    agent_vel_y += GRAVITY
    agent.y += agent_vel_y

    # Set the on-ground state to 0 (not on ground)
    on_ground = 0

    # Check for collisions with the ground
    if agent.colliderect(ground):
        agent.y = ground.y - agent.height
        agent_vel_y = 0
        on_ground = 1
    
    # Check for collisions with platforms
    for platform in platforms:
        # Check if the agent is touching a platform and update its position accordingly
        if agent.colliderect(platform):
            if agent_vel_y > 0:
                agent.y = platform.y - agent.height
                agent_vel_y = 0
                on_ground = 1
            elif agent_vel_y < 0:
                agent.y = platform.y + platform.height
                agent_vel_y = 0

    # Check whether or not the agent is touching an objective
    touched_objective = agent.colliderect(objectives[objective_index])

    # If the agent touches an objective, change the objective index
    if touched_objective:
        new_index = objective_index

        # Randomly select a new objective index and ensure it's different from the current one
        while new_index == objective_index:
            new_index = np.random.randint(0, 3)
        
        # Update the objective index
        objective_index = new_index

    # Clear the screen
    screen.fill(WHITE)
    
    # Draw the agent, ground, and objectives
    pygame.draw.rect(screen, RED, agent)
    pygame.draw.rect(screen, GREEN, ground)
    pygame.draw.rect(screen, BLUE, objectives[objective_index])

    # Draw the platforms
    for platform in platforms:
        pygame.draw.rect(screen, GREEN, platform)

    # Update the display
    pygame.display.flip()
    # Ensure the game runs at the specified frames per second
    clock.tick(FPS)

    # Create the state
    state = (agent.x, agent.y, agent_vel_y, objectives[objective_index].x, objectives[objective_index].y, on_ground)
    
    # Set the reward to 0
    reward = 0.0

    # Reward the agent if it touched an objective
    if touched_objective:
        reward = 20.0

    # Calculate the distance to the objective
    curr_dist = np.sqrt((agent.x - objectives[objective_index].x) ** 2 + (agent.y - objectives[objective_index].y) ** 2)

    # Calculate the distance reward
    dist_reward = (prev_dist - curr_dist)
    # Update the reward with the distance reward
    reward += dist_reward

    # Small time penalty
    reward -= 0.01

    # Return the state, reward, and done
    return state, reward, done

# Reset the environment
def reset():
    # Access global variables
    global DEFAULT_STATE, agent_vel_y, objective_index, on_ground, step_count

    # Reset the agent's position and velocity
    agent.x = DEFAULT_STATE[0]
    agent.y = DEFAULT_STATE[1]
    agent_vel_y = DEFAULT_STATE[2]
    # Reset the objective index
    objective_index = 1
    # Reset the on-ground state
    on_ground = DEFAULT_STATE[5]

    # Reset the step count
    step_count = 0

    # Clear the screen
    screen.fill(WHITE)

    # Draw the agent, ground, and objectives
    pygame.draw.rect(screen, RED, agent)
    pygame.draw.rect(screen, GREEN, ground)
    pygame.draw.rect(screen, BLUE, objectives[objective_index])

    # Draw the platforms
    for platform in platforms:
        pygame.draw.rect(screen, GREEN, platform)

    # Update the display
    pygame.display.flip()

    # Return the default state and done flag
    return DEFAULT_STATE, False