""" Basic PMD """
import time
import os
from enum import IntEnum
import multiprocessing as mp

import numpy as np
import numpy.linalg as la

from pg_termination import mdpmodel
from pg_termination import utils
from pg_termination.logger import BasicLogger

TOL = 1e-10
EPS = 1e-2

def policy_update(pi, psi, eta):
    """ Closed-form solution with KL 

    # TODO: Add other divergences, e.g., Tsallis...
    """
    (n_states, n_actions) = psi.shape

    assert (not np.any(np.isnan(psi))), "Found NaN in psi, quitting program"
    pi *= np.exp(-eta*(psi - np.outer(np.min(psi, axis=1), np.ones(n_actions)))).T
    pi /= np.outer(np.ones(n_actions), np.sum(pi, axis=0))

    assert np.allclose(np.sum(pi,axis=0), 1), "Columns do not sum to 1"

    return True

def _train(settings):
    """
    Natural policy gradient. Implementation based off of 
        - https://papers.nips.cc/paper_files/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf
        - Slide 20 (https://www.andrew.cmu.edu/course/10-703/slides/Lecture_NaturalPolicyGradientsTRPOPPO.pdf)
    """
    seed = settings['seed']

    env = mdpmodel.get_env(settings['env_name'], settings['gamma'], seed)

    if "gridworld" in settings['env_name']:
        with open(os.path.join(settings["log_folder"], "gridworld_target_seed=%d.csv" % seed), "w+") as f:
            f.write("target\n%d" % env.get_target())

    logger = BasicLogger(
        fname=os.path.join(settings["log_folder"], "seed=%d.csv" % seed), 
        keys=["iter", "average value", "advantage gap", "greedy advantage gap"], 
        dtypes=['d', 'f', 'f', 'f']
    )

    pi_0 = np.ones((env.n_actions, env.n_states), dtype=float)/env.n_actions
    pi_t = pi_0
    greedy_pi_t = np.copy(pi_0)
    next_greedy_pi_t = np.copy(pi_0)
    pi_star = None

    s_time = time.time()
    # tolerance for optimality (due to floating point error)
    eps_tol = 1e-14/(1.-env.gamma)
    consecutive_natural_fails = 0
    mu = float(1./env.n_states) * np.ones(env.n_states, dtype=float)

    for t in range(settings["n_iters"]):
        (psi_t, V_t) = env.get_advantage(pi_t)

        # check termination of greedy
        utils.set_greedy_policy(greedy_pi_t, psi_t)
        (greedy_psi_t, greedy_V_t) = env.get_advantage(greedy_pi_t)

        if t <= 9 or (t <= 99 and (t+1) % 5 == 0) or (t+1) % 100 == 0:
            print("Iter %d: f=%.2e (gap=%.2e) (ggap=%.2e)" % (t+1, np.dot(env.rho, V_t), np.max(-psi_t), np.max(-greedy_psi_t)))
        if t <= 99 or ((t+1) % 10 == 0):
            logger.log(t+1, np.dot(env.rho, V_t), np.max(-psi_t), np.max(-greedy_psi_t))
            
        if np.max(-psi_t) < eps_tol:
            print("Terminate at %d: f=%.2e (gap=%.2e)" % (t+1, np.dot(env.rho, V_t), np.max(-psi_t)))
            logger.log(t+1, np.dot(env.rho, V_t), np.max(-psi_t), np.max(-greedy_psi_t))
            pi_star = pi_t
            break
        if np.max(-greedy_psi_t) < eps_tol:
            print("Terminate at %d: gf=%.2e (ggap=%.2e)" % (t+1, np.dot(env.rho, greedy_V_t), np.max(-greedy_psi_t)))
            logger.log(t+1, np.dot(env.rho, greedy_V_t), np.max(-psi_t), np.max(-greedy_psi_t))
            pi_star = greedy_pi_t
            break

        # same termination rule as policy iteration (due to floating point errors)
        (greedy_psi_t, greedy_V_t) = env.get_advantage(greedy_pi_t)
        utils.set_greedy_policy(next_greedy_pi_t, greedy_psi_t)
        if np.allclose(next_greedy_pi_t, greedy_pi_t):
            print("Terminate at %d: greedy-f=%.2e (greedy-gap=%.2e)" % (t+1, np.dot(env.rho, greedy_V_t), np.max(-greedy_psi_t)))
            logger.log(t+1, np.dot(env.rho, greedy_V_t), np.max(-psi_t), np.max(-greedy_psi_t))
            pi_star = greedy_pi_t
            break

        # https://www.jmlr.org/papers/volume22/19-736/19-736.pdf: Lemma 40
        kappa_t = env.get_discounted_visitation(pi_t, mu)
        policy_grad_t = np.multiply(psi_t.T, pi_t)
        policy_grad_t = (1.-env.gamma)**(-1)*kappa_t*policy_grad_t
        eta_t = 1.0
        if not policy_update(pi_t, policy_grad_t.T, eta_t):
            break

    print("Total runtime: %.2fs" % (time.time() - s_time))

    logger.save()

    if pi_star is None: pi_star = pi_t
    with open(os.path.join(settings["log_folder"], "pi_seed=%d.csv" % seed), "w+") as f:
        np.savetxt(f, pi_star, fmt="%1.4e")

    with open(os.path.join(settings["log_folder"], "rho_seed=%d.csv" % seed), "w+") as f:
        rho = env.get_steadystate(pi_star)
        np.savetxt(f, np.atleast_2d(rho).T, fmt="%1.5e")

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

    print("Parallel mode=%s" % parallel)
    if parallel:
        # See: https://stackoverflow.com/questions/15414027/multiprocessing-pool-makes-numpy-matrix-multiplication-slower
        # See also: https://stackoverflow.com/questions/47380366/dramatic-slow-down-using-multiprocess-and-numpy-in-python
        pass

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

        print("Start seed %d" % seed)
        p = mp.Process(target=_train, args=(customized_settings,))
        p.start()
        worker_queue.append(p)

    for p in worker_queue:
        p.join()
