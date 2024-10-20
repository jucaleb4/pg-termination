""" Basic stochastic PMD """
import time
import os
from enum import IntEnum
import multiprocessing as mp

import numpy as np

from pg_termination import wbmdp 
from pg_termination import utils
from pg_termination.logger import BasicLogger

TOL = 1e-10

def policy_update(pi, psi, eta):
    """ Closed-form solution with KL """
    (n_states, n_actions) = psi.shape

    pi *= np.exp(-eta*(psi - np.outer(np.min(psi, axis=1), np.ones(n_actions)))).T
    pi /= np.outer(np.ones(n_actions), np.sum(pi, axis=0))

def _train(settings):
    seed = settings['seed']

    env = wbmdp.get_env(settings['env_name'], settings['gamma'], seed)

    logger = BasicLogger(
        fname=os.path.join(settings["log_folder"], "seed=%d.csv" % seed), 
        keys=["iter", "average value", "advantage gap", "true value", "true advantage gap"], 
        dtypes=['d', 'f', 'f', 'f', 'f']
    )

    pi_0 = np.ones((env.n_actions, env.n_states), dtype=float)/env.n_actions
    pi_t = pi_0
    greedy_pi_t = np.copy(pi_0)
    next_greedy_pi_t = np.copy(pi_0)
    pi_star = None

    eta = settings["eta"]

    s_time = time.time()
    if settings["estimate_Q"] == "linear":
        env.init_estimate_advantage_online_linear(settings)

    for t in range(settings["n_iters"]):
        (true_psi_t, true_V_t) = env.get_advantage(pi_t)

        if settings["estimate_Q"] == "generative":
            (psi_t, V_t) = env.estimate_advantage_generative(pi_t, settings["N"], settings["T"])
        elif settings["estimate_Q"] == "online":
            (psi_t, V_t, _) = env.estimate_advantage_online_mc(pi_t, settings["T"], settings["pi_threshold"])
        elif settings["estimate_Q"] == "linear":
            (psi_t, V_t) = env.estimate_advantage_online_linear(pi_t, settings["T"])
        else: 
            raise Exception("Unknown estimate_Q setting %s" % settings["estimate_Q"])

        if t <= 9 or (t <= 99 and (t+1) % 5 == 0) or (t+1) % 100 == 0:
            print("Iter %d: f=%.2e (gap=%.2e) | true_f=%.2e (true_gap=%.2e)" % (
                t+1, np.mean(V_t), np.max(-psi_t), np.mean(true_V_t), np.max(-true_psi_t)
            ))
        if t <= 99 or ((t+1) % 10 == 0):
            logger.log(t+1, np.mean(V_t), np.max(-psi_t), np.mean(true_V_t), np.max(-true_psi_t))

        policy_update(pi_t, psi_t, eta) 

    print("Total runtime: %.2fs" % (time.time() - s_time))

    logger.save()

def train(settings):
    seed_0 = settings["seed_0"]
    n_seeds = settings["n_seeds"]
    parallel = settings["parallel"]

    try:
        num_workers = min(n_seeds, len(os.sched_getaffinity(0))-1)
        print("Parallel PO experiements with %d workers (%d jobs, %d max cpu)" % (
            num_workers, n_seeds, mp.cpu_count())
        )
    except:
        if 'sched_getffinity' not in dir(os):
            print("Function `os.sched_getaffinity(0)` not available on current OS.\nSetting parallel=False.\nSee https://stackoverflow.com/questions/42658331/python-3-on-macos-how-to-set-process-affinity")
            parallel = False

    worker_queue = []

    for seed in range(seed_0, seed_0+n_seeds):
        customized_settings = settings.copy()
        customized_settings["seed"] = seed

        if not parallel:
            _train(customized_settings)
            continue

        if len(worker_queue) == num_workers:
            # wait for all workers to finish
            for p in worker_queue:
                p.join()
            worker_queue = []
        p = mp.Process(target=_train, args=(customized_settings,))
        p.start()
        worker_queue.append(p)

