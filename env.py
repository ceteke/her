import numpy as np


class Env():
    def __init__(self, size):
        self.size = size

    def reset(self):
        state = np.random.randint(2, size=self.size)
        goal = np.random.randint(2, size=self.size)

        while np.sum(state == goal) == self.size:
            goal = np.random.randint(2, size=self.size)
        self.goal = goal
        return state

    def step(self, state, action):
        next_state = np.copy(state)
        next_state[action] = 1 - next_state[action]

        if not np.sum(next_state == self.goal) == self.size:
            return next_state, -1, False
        else:
            return next_state, +1, True