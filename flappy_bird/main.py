import pygame
import sys
from obstacle import ObstacleGenerator
from environment import Environment
from agent import Agent
from experience_replay import ExperienceReplay

pygame.init()
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

obstacle_generator = ObstacleGenerator(screen, obstacle_width=75)
env = Environment(screen=screen, render_on=True, obstacle_generator=obstacle_generator)
agent = Agent(epsilon=0.001, epsilon_decay=0.995, epsilon_end=0.001)
experience_replay = ExperienceReplay(capacity=10000, batch_size=32)

agent.load('models/model.h5')

episodes = 5000

for episode in range(episodes):
    state = env.reset()
    step = 0

    done = False
    while not done:
        step+=1
        # print()
        # print('Obstacle Locations:', [obstacle.horizontal_location for obstacle in obstacle_generator.obstacles])
        # print('Episode:', episode)
        # print('Step:', step)
        # print('Epsilon:', agent.epsilon)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        action = agent.get_action(state)

        reward, next_state, done = env.step(action)

        # experience_replay.add_experience(state, action, reward, next_state, done)

        # # If the experience replay has enough memory to provide a sample, train the agent
        # if experience_replay.can_provide_sample():
        #     experiences = experience_replay.sample_batch()
        #     agent.learn(experiences)

        # Set the state to the next_state
        state = next_state
    
    agent.save('models/model.h5')

pygame.quit()
sys.exit()
