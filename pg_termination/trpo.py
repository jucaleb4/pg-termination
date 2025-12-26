""" Basic PMD """
import time
import os
import math
from enum import IntEnum
import multiprocessing as mp

import numpy as np
import numpy.linalg as la

from pg_termination import mdpmodel 
from pg_termination import utils
from pg_termination.logger import BasicLogger

TOL = 1e-10
EPS = 1e-1

def policy_update(pi, psi, eta):
    """ Closed-form solution with KL 

    # TODO: Add other divergences, e.g., Tsallis...
    """
    (n_states, n_actions) = psi.shape

    assert (not np.any(np.isnan(psi))), "Found NaN in psi, quitting program"
    pi *= np.exp(-eta*(psi - np.outer(np.min(psi, axis=1), np.ones(n_actions)))).T
    pi_new = pi / np.outer(np.ones(n_actions), np.sum(pi, axis=0))

    assert np.allclose(np.sum(pi_new,axis=0), 1), "Columns do not sum to 1"

    return pi_new

def _train(settings):
    """
    Trust region policy optimization. Implementation based off of 
        - https://proceedings.mlr.press/v37/schulman15.pdf
        - Slide 24 (https://www.andrew.cmu.edu/course/10-703/slides/Lecture_NaturalPolicyGradientsTRPOPPO.pdf)
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
    eps = EPS

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

        # pi_t_flat = np.reshape(pi_t.T, newshape=(-1,))
        # n_A = pi_t.shape[0]
        nu_t = env.get_stationary(pi_t)
        # nu_t_ext = np.kron(nu_t, np.ones(n_A))
        # F_t = np.diag(np.multiply(nu_t_ext, pi_t_flat))
        # F_t -= np.outer(pi_t_flat, np.multiply(nu_t_ext, np.power(pi_t_flat, 3)))
        # F_t -= (np.outer(pi_t_flat, np.multiply(nu_t_ext, np.power(pi_t_flat, 3)))).T
        # F_t += np.dot(nu_t_ext, np.power(pi_t_flat,3)) * np.outer(pi_t_flat, pi_t_flat)
        # psi_t_flat = np.reshape(psi_t, newshape=(-1,))
        # natural_psi_t = np.reshape(la.solve(F_t, psi_t_flat), newshape=psi_t.shape)

        # # line search
        # eta_t = eta_t_0 = 1./la.norm(natural_psi_t)
        # pi_new = policy_update(pi_t, natural_psi_t, eta_t)
        # loss = np.dot(nu_t, np.einsum("sa,as->s", psi_t, pi_new))
        # KL_dist = np.dot(nu_t, np.einsum("as,as->s", pi_new, np.log(np.divide(pi_new, pi_t))))
        eta_t = 1.
        pi_new = policy_update(pi_t, psi_t, eta_t)
        loss = np.dot(nu_t, np.einsum("sa,as->s", psi_t, pi_new))
        KL_dist = np.dot(nu_t, np.einsum("as,as->s", pi_new, np.divide(pi_new + 1e-12, pi_t + 1e-12)))

        exponential_inc = loss <= 0 and KL_dist <= eps
        eta_scale = 2.0 if exponential_inc else 0.5
        # only allow binary search of 10
        for _ in range(10):
            eta_t *= eta_scale
            pi_new = policy_update(pi_t, psi_t, eta_t)
            loss = np.dot(nu_t, np.einsum("sa,as->s", psi_t, pi_new))
            KL_dist = np.dot(nu_t, np.einsum("as,as->s", pi_new, np.divide(pi_new + 1e-12, pi_t + 1e-12)))

            # check if we can continue line search
            if exponential_inc and (not (loss <= 0 and KL_dist <= eps)):
                eta_t /= 2
                pi_new = policy_update(pi_t, psi_t, eta_t)
                break
            if (not exponential_inc) and (loss <= 0 and KL_dist <= eps):
                break

        pi_t = pi_new

        # except:
        #     print(">>> F_t is singular, trying least squares")
        #     natural_psi_t = np.reshape(la.lstsq(F_t, psi_t_flat)[0], newshape=psi_t.shape)
        # if np.max(np.abs(natural_psi_t) > 1e9):
        #     print(">>> natural psi_t is too large, resorting to normal policy gradient")
        #     natural_psi_t = psi_t
        #     consecutive_natural_fails += 1
        # else:
        #     consecutive_natural_fails = 0
        # else:
        #     # if too many consecutive fails, resort back to PMD
        #     natural_psi_t = psi_t

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
