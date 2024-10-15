import time
import os

import numpy as np

from wbmdp import GridWorldWithTraps
from wbmdp import Taxi
from wbmdp import Random
from wbmdp import Small
from wbmdp import Chain
from wbmdp import GridWorldWithTrapsAndHills

from utils import set_greedy_policy

from logger import BasicLogger

def main():
    args = dict({
        "n_iters": 10000,
        "fname": os.path.join("logs", "gridworld_policyiter.csv")
    })
    # env = GridWorldWithTraps(20, 20, 0.99, seed=1104, ergodic=True)
    env = GridWorldWithTrapsAndHills(30, 30, 0.5, seed=1104, ergodic=True)
    # env = Taxi(0.995, ergodic=True)
    # env = Random(100, 100, 0.996, seed=1101)
    # env = Small(100, 0.99995, eps=1e-8, seed=1104)
    # env = Chain(100, 0.995, eps=1e-3, seed=1104)

    logger = BasicLogger(
        fname=args["fname"], 
        keys=["iter", "average value", "advantage gap"], 
        dtypes=['d', 'f', 'f']
    )

    pi_0 = np.zeros((env.n_actions, env.n_states), dtype=float)
    pi_0[0,:] = 1.
    pi_t = pi_0
    next_pi_t = np.copy(pi_0)

    s_time = time.time()

    for t in range(args["n_iters"]):
        (psi_t, V_t) = env.get_advantage(pi_t)

        # check termination of greedy
        set_greedy_policy(next_pi_t, psi_t)

        if t <= 9 or (t <= 99 and (t+1) % 5 == 0) or (t+1) % 100 == 0:
            print("Iter %d: f=%.2e (gap=%.2e)" % (t+1, np.mean(V_t), np.max(-psi_t)))
        if t <= 99 or ((t+1) % 10 == 0):
            logger.log(t+1, np.mean(V_t), np.max(-psi_t))

        if np.allclose(next_pi_t, pi_t):
            print("Terminate at %d: f=%.2e (gap=%.2e)" % (t+1, np.mean(V_t), np.max(-psi_t)))
            logger.log(t+1, np.mean(V_t), np.max(-psi_t))
            break

        pi_t[:len(pi_t),:] = next_pi_t

    print("Total runtime: %.2fs" % (time.time() - s_time))

    logger.save()

if __name__ == '__main__':
    main()
