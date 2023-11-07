from environment import Environment
from agent import Agent
from experience_replay import ExperienceReplay
import time

if __name__ == '__main__':

    grid_size = 5
    environment = Environment(grid_size=grid_size, render_on=True)
    agent = Agent(grid_size=grid_size, epsilon=1, epsilon_decay=0.998, epsilon_end=0.01)
    agent.load('models/model')

    experience_replay = ExperienceReplay(capacity=2000, batch_size=32)
    
    # Number of episodes to run before training stops
    episodes = 10000

    for episode in range(episodes):
        # Get the initial state of the environment and set done to False
        state = environment.reset()
        done = False
        
        # Loop until the episode finishes
        while not done:

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
            # time.sleep(0.5)

        print(episode)
        agent.save(f'models/model')
