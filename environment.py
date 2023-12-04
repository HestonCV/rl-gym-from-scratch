import random
import numpy as np

class Environment:
    def __init__(self, grid_size, render_on=False):
        self.grid_size = grid_size
        self.grid = []
        self.render_on = render_on
        self.agent_location = None
        self.goal_location = None

    def reset(self):
        # Initialize the empty grid as a 2d array of 0s
        self.grid = np.zeros((self.grid_size, self.grid_size))

        # Add the agent and the goal to the grid
        self.agent_location = self.add_agent()
        self.goal_location = self.add_goal()

        if self.render_on:
            self.render()

        # Return the initial state of the grid
        return self.get_state()

    def add_agent(self):
        # Choose a random location
        location = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        
        # Agent is represented by a 1
        self.grid[location[0]][location[1]] = 1
        
        return location

    def add_goal(self):
        # Choose a random location
        location = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))

        # Get a random location until it is not occupied
        while self.grid[location[0]][location[1]] == 1:
            location = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        
        # Goal is represented by a -1
        self.grid[location[0]][location[1]] = -1

        return location

    def render(self):
        # Convert to a list of ints to improve formatting
        grid = self.grid.astype(int).tolist()

        for row in grid:
            print(row)
        print('') # To add some space between renders for each step
    
    def get_state(self):
        # Flatten the grid from 2d to 1d
        state = self.grid.flatten()
        return state

    def move_agent(self, action):
        # Map agent action to the correct movement
        moves = {
            0: (-1, 0), # Up
            1: (1, 0),  # Down
            2: (0, -1), # Left
            3: (0, 1)   # Right
        }
        
        previous_location = self.agent_location
        
        # Determine the new location after applying the action
        move = moves[action]
        new_location = (previous_location[0] + move[0], previous_location[1] + move[1])
        
        done = False # The episode is not done by default
        reward = 0   # Initialize reward
            
        # Check for a valid move
        if self.is_valid_location(new_location):
            # Remove agent from old location
            self.grid[previous_location[0]][previous_location[1]] = 0
            
            # Add agent to new location
            self.grid[new_location[0]][new_location[1]] = 1
            
            # Update agent's location
            self.agent_location = new_location
            
            # Check if the new location is the reward location
            if self.agent_location == self.goal_location:
                # Reward for getting the goal
                reward = 100
                
                # Episode is complete
                done = True
            else:
                # Calculate the distance before the move
                previous_distance = np.abs(self.goal_location[0] - previous_location[0]) + \
                                    np.abs(self.goal_location[1] - previous_location[1])
                        
                # Calculate the distance after the move
                new_distance = np.abs(self.goal_location[0] - new_location[0]) + \
                               np.abs(self.goal_location[1] - new_location[1])
                
                # If new_location is closer to the goal, reward = 1, if further, reward = -1
                reward = (previous_distance - new_distance) - 0.1
        else:
            # Slightly larger punishment for an invalid move
            reward = -3
        
        return reward, done
    
    def is_valid_location(self, location):
        # Check if the location is within the boundaries of the grid
        if (0 <= location[0] < self.grid_size) and (0 <= location[1] < self.grid_size):
            return True
        else:
            return False
        
    def step(self, action):
        # Apply the action to the environment, record the observations
        reward, done = self.move_agent(action)
        next_state = self.get_state()
    
        # Render the grid at each step
        if self.render_on:
            self.render()
    
        return reward, next_state, done