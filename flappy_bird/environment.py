import random
import numpy as np
import pygame

class Environment:
    def __init__(self, screen, obstacle_generator, agent_size=50, render_on=False):
        self.screen = screen
        self.screen_width, self.screen_height = screen.get_size()
        self.render_on = render_on
        self.obstacle_generator = obstacle_generator
        self.agent_size = agent_size
        self.agent_velocity = 0
        self.gravity = 2

        self.agent_vertical_location = (self.screen_height / 2) - (self.agent_size / 2)
        self.agent_horizontal_location = (self.screen_width / 2) - (self.agent_size / 2)

        self.agentRect = pygame.Rect(self.agent_horizontal_location, self.agent_vertical_location, self.agent_size, self.agent_size)

    def reset(self):
        self.agent_vertical_location = (self.screen_height / 2) - (self.agent_size / 2)
        self.agent_velocity = 0

        self.obstacle_generator.reset()

        if self.render_on:
            self.render()

        return self.get_state()

    def render(self):
        self.screen.fill((0, 200, 255))
        self.obstacle_generator.render()
        pygame.draw.rect(self.screen, (255, 0, 0), self.agentRect)
        pygame.display.flip()
    
    def get_state(self):
        obstacle_distance, obstacle_opening_location = self.obstacle_generator.get_closest_obstacle(self.agent_horizontal_location)

        state = np.array([obstacle_distance, obstacle_opening_location, self.agent_vertical_location, self.agent_velocity])
        return state

    def move_agent(self, action):
        if action == 1:
            if self.agent_velocity > -20:
                self.agent_velocity -= 7

        done = False
        reward = 0.1

        if self.agent_velocity < 5:
            self.agent_velocity += self.gravity
            
        self.agent_vertical_location += self.agent_velocity

        if self.agent_vertical_location <= 0:
            done = True
            reward = -1

        elif self.agent_vertical_location >= 600 - self.agent_size:
            done = True
            reward = -1

        for obstacle in self.obstacle_generator.obstacles:
            if self.agentRect.colliderect(obstacle.top_rect) or self.agentRect.colliderect(obstacle.bottom_rect):
                done = True
                reward = -1

        _, closest_opening_location = self.obstacle_generator.get_closest_obstacle(self.agent_horizontal_location)

        if closest_opening_location - 30 <= self.agent_vertical_location <= closest_opening_location + 30:
            reward += 0.2
            reward = round(reward, 1)

        # Update the Rect position
        self.agentRect.y = self.agent_vertical_location
        

        return reward, done
    
        
    def step(self, action):
        # Apply the action to the environment, record the observations
        reward, done = self.move_agent(action)
        print('reward', reward)
        next_state = self.get_state()

        self.obstacle_generator.step()
    
        # Render the grid at each step
        if self.render_on:
            self.render()
    
        return reward, next_state, done