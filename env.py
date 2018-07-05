import numpy as np


class Env():
    def __init__(self, size):
        self.size = size

    def reset(self):
        state = np.random.randint(2, size=self.size)
        target = np.random.randint(2, size=self.size)

        while np.sum(state == target) == self.size:
            target = np.random.randint(2, size=self.size)
        self.target = target
        return state

    def step(self, state, action):
        next_state = np.copy(state)
        next_state[action] = 1 - next_state[action]

        if not np.sum(next_state == self.target) == self.size:
            return next_state, -1
        else:
            return next_state, 0