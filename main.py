from environment import Environment
from agent import Agent
from experience_replay import ExperienceReplay
import time

if __name__ == '__main__':

    grid_size = 8

    environment = Environment(grid_size=grid_size, render_on=True)
    agent = Agent(grid_size=grid_size, epsilon=1, epsilon_decay=0.9998, epsilon_end=0.01)
    experience_replay = ExperienceReplay(capacity=10000, batch_size=32)
    # agent.load(f'models/model_{grid_size}.h5')
    
    # Number of episodes to run before training stops
    episodes = 5000
    # Max number of steps in each episode
    max_steps = 250

    for episode in range(episodes):
        # Get the initial state of the environment and set done to False
        state = environment.reset()

        # Loop until the episode finishes
        for step in range(max_steps):
            print('Episode:', episode)
            print('Step:', step)
            print('Epsilon:', agent.epsilon)

            # Get the action choice from the agents policy
            action = agent.get_action(state)

            # Take a step in the environment and save the experience
            reward, next_state, done = environment.step(action)
            experience_replay.add_experience(state, action, reward, next_state, done)

            # If the experience replay has enough memory to provide a sample, train the agent
            if experience_replay.can_provide_sample():
                experiences = experience_replay.sample_batch()
                agent.learn(experiences)

            # Set the state to the next_state
            state = next_state

            if done:
                break

            # Optionally, pause for half a second to evaluate the model
            # time.sleep(0.5)
    
        agent.save(f'models/model_{grid_size}.h5')