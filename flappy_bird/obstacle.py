import pygame
import random

class Obstacle:
    def __init__(self, screen, width, opening_location):
        self.screen = screen
        self.horizontal_location = 800
        self.opening_location = opening_location
        self.width = width
        self.top_height = (600 * (self.opening_location / 600) - 75)
        self.bottom_height = (600 * (1 - (self.opening_location / 600)) - 75)
        self.top_rect = pygame.Rect(self.horizontal_location, 0, 50, self.top_height)
        self.bottom_rect = pygame.Rect(self.horizontal_location, 600 - self.bottom_height, 50, self.bottom_height)

    def draw(self):
        pygame.draw.rect(self.screen, (0, 255, 0), (self.horizontal_location, 0, 50, self.top_height))
        pygame.draw.rect(self.screen, (0, 255, 0), (self.horizontal_location, 600 - self.bottom_height, 50, self.bottom_height))


class ObstacleGenerator:
    def __init__(self, screen, obstacle_width):
        self.screen = screen
        self.obstacle_width = 50
        self.obstacles = []
        self.steps = 0
        self.speed = 4
        self.obstacle_interval = 80
    
    def make_obstacle(self):
        opening_location = random.randint(100, 500)
        obstacle = Obstacle(self.screen, self.obstacle_width, opening_location)
        self.obstacles.append(obstacle)
    
    def get_closest_obstacle(self, agent_horizontal_location):
        closest_obstacle = None
        closest_distance = float('inf')

        for obstacle in self.obstacles:
            distance = obstacle.horizontal_location - agent_horizontal_location + self.obstacle_width

            if distance >= 0 and distance <= closest_distance:
                closest_obstacle = obstacle
                closest_distance = distance

        obstacle_distance = closest_distance
        obstacle_opening_location = closest_obstacle.opening_location

        return obstacle_distance, obstacle_opening_location
    
    def step(self):
        self.steps += 1
        for obstacle in self.obstacles[:]:

            if obstacle.horizontal_location <= -self.obstacle_width:
                self.obstacles.pop(0)
                continue

            obstacle.horizontal_location = obstacle.horizontal_location - self.speed

            # Update the positions of the top and bottom rects
            obstacle.top_rect.x = obstacle.horizontal_location
            obstacle.bottom_rect.x = obstacle.horizontal_location

            obstacle.draw()
        
        if self.steps % self.obstacle_interval == 0:
            self.make_obstacle()
        
    def reset(self):
        self.steps = 0
        self.obstacles = []
        self.make_obstacle()


    def render(self):
        for obstacle in self.obstacles:
            obstacle.draw()