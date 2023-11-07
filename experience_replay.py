import random
from collections import deque, namedtuple

class ExperienceReplay:
    def __init__(self, capacity, batch_size):
        # Memory stores the experiences in a deque, so if capacity is exceeded it removes
        # the oldest item efficiently
        self.memory = deque(maxlen=capacity)

        # Batch size specifices the amount of experiences that will be sampled at once
        self.batch_size = batch_size

        # Experience is a namedtuple that stores the relevant information for training
        self.Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

    def add_experience(self, state, action, reward, next_state, done):
        # Create a new experience and store it in memory
        experience = self.Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample_batch(self):
        # Batch will be a random sample of experiences from memory of size batch_size
        batch = random.sample(self.memory, self.batch_size)
        return batch

    def can_provide_sample(self):
        # Determines if the length of memory has exceeded batch_size
        return len(self.memory) >= self.batch_size