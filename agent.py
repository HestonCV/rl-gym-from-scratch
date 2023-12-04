from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
import numpy as np   

class Agent:
    def __init__(self, grid_size, epsilon=1, epsilon_decay=0.998, epsilon_end=0.01, gamma=0.99):
        self.grid_size = grid_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.gamma = gamma
        self.model = self.build_model()

    def build_model(self):
        # Create a sequential model with 3 layers
        model = Sequential([
            # Input layer expects a flattened grid, hence the input shape is grid_size squared
            Dense(128, activation='relu', input_shape=(self.grid_size**2,)),
            Dense(64, activation='relu'),
            # Output layer with 4 units for the possible actions (up, down, left, right)
            Dense(4, activation='linear')
        ])

        model.compile(optimizer='adam', loss='mse')

        return model 

    def get_action(self, state):
        # rand() returns a random value between 0 and 1
        if np.random.rand() <= self.epsilon:
            # Exploration: random action
            action = np.random.randint(0, 4)
        else:
            # Add an extra dimension to the state to create a batch with one instance
            state = np.expand_dims(state, axis=0)

            # Use the model to predict the Q-values (action values) for the given state
            q_values = self.model.predict(state, verbose=0)

            # Select and return the action with the highest Q-value
            action = np.argmax(q_values[0]) # Take the action from the first (and only) entry
        
        # Decay the epsilon value to reduce the exploration over time
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return action
    

    def learn(self, experiences):
        states = np.array([experience.state for experience in experiences])
        actions = np.array([experience.action for experience in experiences])
        rewards = np.array([experience.reward for experience in experiences])
        next_states = np.array([experience.next_state for experience in experiences])
        dones = np.array([experience.done for experience in experiences])

        # Predict the Q-values (action values) for the given state batch
        current_q_values = self.model.predict(states, verbose=0)

        # Predict the Q-values for the next_state batch
        next_q_values = self.model.predict(next_states, verbose=0)

                # Initialize the target Q-values as the current Q-values
        target_q_values = current_q_values.copy()

        # Loop through each experience in the batch
        for i in range(len(experiences)):
            if dones[i]:
                # If the episode is done, there is no next Q-value
                # [i, actions[i]] is the numpy equivalent of [i][actions[i]]
                target_q_values[i, actions[i]] = rewards[i]
            else:
                # The updated Q-value is the reward plus the discounted max Q-value for the next state
                # [i, actions[i]] is the numpy equivalent of [i][actions[i]]
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
            
        # Train the model
        self.model.fit(states, target_q_values, epochs=1, verbose=0)

    def load(self, file_path):
        self.model = load_model(file_path)

    def save(self, file_path):
        self.model.save(file_path)