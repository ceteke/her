from env import Env
from dqn import DQN
import argparse
import random


parser = argparse.ArgumentParser(description='HER Bit Flipping')
parser.add_argument('-v', action='store_true', help='Verbose flag')
parser.add_argument('-s', type=int, default=15, help='Size of bit string')
parser.add_argument('-i', type=int, default=200, help='Num epochs')
parser.add_argument('-e', type=int, default=16, help='Num episodes')
parser.add_argument('-c', type=int, default=50, help='Num cycles')
parser.add_argument('-o', type=int, default=40, help='Optimization steps')

args = parser.parse_args()

her = True

dqn = DQN(args.s)
env = Env(args.s)

for epoch in range(args.i):
    num_success = 0
    print("Epoch {}".format(epoch+1))
    for cycle in range(args.c):
        for episode in range(args.e):
            state = env.reset()
            experience = []

            # Actually take steps
            for t in range(args.s):
                action = dqn.get_action(state, env.goal)
                next_state, reward, done = env.step(state, action)
                dqn.store_transition(state, action, reward, next_state, env.goal)
                experience.append((state, action, reward, next_state))
                if done:
                    num_success += 1
                    break
                state = next_state

            # Change the goals and take steps
            if her:
                exp_len = len(experience)
                for t in range(exp_len):
                    s_t, a_t, _, sn_t = experience[t]
                    for g in range(4):
                        virtual_goal = experience[random.randint(t, exp_len-1)][3]
                        success = env.check_success(sn_t, virtual_goal)
                        r_t = 0 if success else -1
                        dqn.store_transition(s_t, a_t, r_t, sn_t, virtual_goal)

        for opt_step in range(args.o):
            dqn.update()
    print("Success: {0:.2f}%".format((num_success/(args.c*args.e))*100))