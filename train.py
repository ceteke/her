from env import Env
from dqn import DQN
import argparse


parser = argparse.ArgumentParser(description='HER Bit Flipping')
parser.add_argument('-v', action='store_true', help='Verbose flag')
parser.add_argument('-s', type=int, default=10, help='Size of bit string')
parser.add_argument('-i', type=int, default=200, help='Num epochs')
parser.add_argument('-e', type=int, default=16, help='Num episodes')
parser.add_argument('-c', type=int, default=50, help='Num cycles')
parser.add_argument('-o', type=int, default=40, help='Optimization steps')

args = parser.parse_args()

dqn = DQN(args.s)
env = Env(args.s)

for epoch in range(args.i):
    print("Epoch {}".format(epoch+1))
    for cycle in range(args.c):
        for episode in range(args.e):
            state = env.reset()
            for t in range(args.s):
                action = dqn.get_action(state, env.goal)
                next_state, reward = env.step(state, action)
                dqn.store_transition(state, action, reward, next_state, env.goal)
        for opt_step in range(args.o):
            dqn.update()