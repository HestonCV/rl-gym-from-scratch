import random

class Environment:
    def __init__(self, grid_size, render_on=False):
        self.grid_size = grid_size
        self.render_on = render_on
        self.grid = []
        self.agent_location = None
        self.goal_location = None

    def reset(self):
        # Initialize the empty grid as a 2d list of 0s
        self.grid = [[0] * self.grid_size for _ in range(self.grid_size)]

        # Add the agent and the goal to the grid
        self.agent_location = self.add_agent()
        self.goal_location = self.add_goal()

        if self.render_on:
            self.render()

        # Return the initial state
        return self.get_state()

    def add_agent(self):
        # Agent is represented by a 1
        location = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        self.grid[location[0]][location[1]] = 1
        return location

    def add_goal(self):
        # Goal is represented by a -1
        location = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))

        # Get a random location until it is not occupied
        while self.grid[location[0]][location[1]] == 1:
            location = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))

        self.grid[location[0]][location[1]] = -1

        return location

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

        # Initialize the reward and done flag
        reward = -0.1 # Default penalty for a move
        done = False  # The episode is not done by default

        # Check for a valid move and if the new location is the reward
        if self.is_valid_location(new_location):
            # Update grid with the new agent location
            self.grid[previous_location[0]][previous_location[1]] = 0
            self.grid[new_location[0]][new_location[1]] = 1

            # Check if the new location is the reward location
            if new_location == self.goal_location:
                reward = 1  # The reward for finding the goal
                done = True # The episode is done because the goal is reached
            else:
                reward = -0.1 # The penalty for a move that is not the goal
            
            # Update agent's location
            self.agent_location = new_location
        
        else:
            # If the move is invalid, the agent stays in place, and a larger penalty is given
            reward = -0.3
        
        return reward, done
            
    def get_state(self):
        # Use list comprehension to flatten the grid for use by the model
        state = [cell for row in self.grid for cell in row]
        return state

    def is_valid_location(self, location):
        # Check if the location is within the boundaries of the grid
        if (0 <= location[0] < self.grid_size) and (0 <= location[1] < self.grid_size):
            return True
        else:
            return False
        
    def render(self):
        for row in self.grid:
            print(row)
        print('')

    def step(self, action):
        # Apply the action to the environment return the observations
        reward, done = self.move_agent(action)
        next_state = self.get_state()

        if self.render_on:
            self.render()

        return reward, next_state, done