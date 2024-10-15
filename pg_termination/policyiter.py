import time
import os

import numpy as np

from pg_termination import wbmdp 
from pg_termination import utils
from pg_termination.logger import BasicLogger

def _train(settings):
    seed = settings['seed']

    env = wbmdp.get_env(settings['env_name'], settings['gamma'], seed)

    logger = BasicLogger(
        fname=os.path.join(settings["log_folder"], "seed=%d.csv" % seed), 
        keys=["iter", "average value", "advantage gap"], 
        dtypes=['d', 'f', 'f']
    )

    pi_0 = np.zeros((env.n_actions, env.n_states), dtype=float)
    pi_0[0,:] = 1.
    pi_t = pi_0
    next_pi_t = np.copy(pi_0)

    s_time = time.time()

    for t in range(settings["n_iters"]):
        (psi_t, V_t) = env.get_advantage(pi_t)

        # check termination of greedy
        utils.set_greedy_policy(next_pi_t, psi_t)

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
    with open(os.path.join(settings["log_folder"], "pi_seed=%d.csv" % seed), "w+") as f:
        np.savetxt(f, pi_t, fmt="%1.4e")

    logger.save()

def train(settings):
    seed_0 = settings["seed_0"]
    n_seeds = settings["n_seeds"]

    for seed in range(seed_0, seed_0+n_seeds):
        settings["seed"] = seed
        _train(settings)
