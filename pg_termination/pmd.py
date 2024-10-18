""" Basic PMD """
import time
import os
from enum import IntEnum
import multiprocessing as mp

import numpy as np

from pg_termination import wbmdp 
from pg_termination import utils
from pg_termination.logger import BasicLogger

TOL = 1e-10

class StepSize(IntEnum):
    SUBLINEAR = 0
    KL_LINEAR_GEOMETRIC = 1
    KL_LINEAR_ADAPTIVE = 2
    EUCLIDEAN_LINEAR_ADAPTIVE = 3

class Update(IntEnum):
    EUCLIDEAN_UPDATE = 100
    KL_UPDATE = 101

def simplex_projection(x):
    """ Strongly polynomial time for projecting onto simplex.

    Paper: https://arxiv.org/pdf/1101.6081

    """
    n_actions = len(x)
    x_sorted_index = np.argsort(x)
    x_sorted_index_inverse = np.argsort(x_sorted_index)
    x_sorted = x[x_sorted_index]
    t = np.divide(np.cumsum(x_sorted[::-1])-1, np.arange(1,n_actions+1))[::-1]
    x_sorted_aug = np.append(t[0], x_sorted)
    i = np.argmax(np.logical_and(x_sorted_aug[:-1] <= t, t <= x_sorted_aug[1:]))
    x_sorted_proj = np.maximum(0, x_sorted-t[i])
    x_proj = x_sorted_proj[x_sorted_index_inverse]

    assert abs(np.sum(x_proj) - 1) <= TOL, "Sum of sum(x)=%.4e != 1.0" % np.sum(x_proj)

    return x_proj

def parallel_simplex_projection(X):
    """ Parallel version of simplex_projection. Project each row of X """
    (n_actions, n_states) = X.shape
    X_sorted_index = np.argsort(X, axis=0)
    X_sorted_index_inverse = np.argsort(X_sorted_index, axis=0)
    X_sorted = X[X_sorted_index, np.arange(n_states)]
    T = np.divide(np.cumsum(X_sorted[::-1], axis=0)-1, np.outer(np.arange(1,n_actions+1), np.ones(n_states)))[::-1]
    X_sorted_aug = np.vstack((T[0,:], X_sorted))
    I = np.argmax(np.logical_and(X_sorted_aug[:-1,:] <= T, T <= X_sorted_aug[1:,:]), axis=0)
    X_sorted_proj = np.maximum(0, X_sorted-np.outer(np.ones(n_actions), T[I, np.arange(n_states)], ))
    X_proj = X_sorted_proj[X_sorted_index_inverse, np.arange(n_states)]

    assert np.max(np.abs(np.sum(X_proj, axis=0) - 1)) <= TOL, "Sum of sum(x)=%.11e != 1.0" % np.max(np.sum(X_proj, axis=0))

    return X_proj

def policy_update(pi, psi, eta, update_rule):
    """ Projection onto simplex with Euclidean or closed-form solution with KL """
    (n_states, n_actions) = psi.shape

    if update_rule == Update.EUCLIDEAN_UPDATE:
        pi_gd = pi - eta*psi.T
        pi[:] = parallel_simplex_projection(pi_gd)
    elif update_rule == Update.KL_UPDATE:
        pi *= np.exp(-eta*(psi - np.outer(np.min(psi, axis=1), np.ones(n_actions)))).T
        pi /= np.outer(np.ones(n_actions), np.sum(pi, axis=0))

class StepsizeSchedule():
    """ Returns steps size """
    def __init__(self, env, stepsize_rule, eta):
        """
        :param env: environment
        :param stepsize_rule: which stepsize we want to use
        :param eta: stepsize for constant (i.e., sublinear) setting
        """
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.gamma = env.gamma
        self.stepsize_rule = stepsize_rule

        self.eta = eta
        self.N = np.ceil(4./(1-self.gamma))
        self.T = np.ceil(np.log(self.n_states**3*self.n_actions/(1.-self.gamma)**2/np.log(2)))
        self.Delta = 0

        if stepsize_rule in [StepSize.EUCLIDEAN_LINEAR_ADAPTIVE]:
            print("Worst case iteration complexity: %d" % (self.n_states*(self.n_actions-1)*self.N*self.T))

    def get_stepsize(self, t, psi):
        if self.stepsize_rule == StepSize.SUBLINEAR:
            return self.eta
        elif self.stepsize_rule == StepSize.KL_LINEAR_GEOMETRIC:
            return (1./self.gamma)**t
        elif self.stepsize_rule == StepSize.KL_LINEAR_ADAPTIVE:
            (_, n_actions) = psi.shape
            # dynamically update
            if t % (self.N * self.T) == 0:
                self.Delta = np.max(-psi)/(1.-self.gamma)
            eta = np.log(n_actions)*4**(np.floor(t/self.N) % self.T)*2**(self.T*np.floor(t/(self.T*self.N)))/self.Delta
            return eta
        elif self.stepsize_rule == StepSize.EUCLIDEAN_LINEAR_ADAPTIVE:
            # dynamically update
            if t % (self.N * self.T) == 0:
                self.Delta = np.max(-psi)/(1.-self.gamma)
            eta = 2**(1+np.floor(t/self.N) % self.T)/self.Delta
            return eta

def _train(settings):
    seed = settings['seed']

    env = wbmdp.get_env(settings['env_name'], settings['gamma'], seed)

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

    stepsize_scheduler = StepsizeSchedule(env, settings["stepsize_rule"], settings.get("eta",1))
    s_time = time.time()
    # tolerance for optimality (due to floating point error)
    eps_tol = 1e-12

    for t in range(settings["n_iters"]):
        (psi_t, V_t) = env.get_advantage(pi_t)

        # check termination of greedy
        utils.set_greedy_policy(greedy_pi_t, psi_t)
        (greedy_psi_t, greedy_V_t) = env.get_advantage(greedy_pi_t)

        if t <= 9 or (t <= 99 and (t+1) % 5 == 0) or (t+1) % 100 == 0:
            print("Iter %d: f=%.2e (gap=%.2e) (ggap=%.2e)" % (t+1, np.mean(V_t), np.max(-psi_t), np.max(-greedy_psi_t)))
        if t <= 99 or ((t+1) % 10 == 0):
            logger.log(t+1, np.mean(V_t), np.max(-psi_t), np.max(-greedy_psi_t))
            
        if np.max(-psi_t) < eps_tol:
            print("Terminate at %d: f=%.2e (gap=%.2e)" % (t+1, np.mean(V_t), np.max(-psi_t)))
            logger.log(t+1, np.mean(V_t), np.max(-psi_t), np.max(-greedy_psi_t))
            pi_star = pi
            break
        if np.max(-greedy_psi_t) < eps_tol:
            print("Terminate at %d: gf=%.2e (ggap=%.2e)" % (t+1, np.mean(greedy_V_t), np.max(-greedy_psi_t)))
            logger.log(t+1, np.mean(greedy_V_t), np.max(-psi_t), np.max(-greedy_psi_t))
            pi_star = greedy_pi_t
            break

        # same termination rule as policy iteration (due to floating point errors)
        (greedy_psi_t, greedy_V_t) = env.get_advantage(greedy_pi_t)
        utils.set_greedy_policy(next_greedy_pi_t, greedy_psi_t)
        if np.allclose(next_greedy_pi_t, greedy_pi_t):
            print("Terminate at %d: greedy-f=%.2e (greedy-gap=%.2e)" % (t+1, np.mean(greedy_V_t), np.max(-greedy_psi_t)))
            logger.log(t+1, np.mean(greedy_V_t), np.max(-psi_t), np.max(-greedy_psi_t))
            pi_star = greedy_pi_t
            break

        eta_t = stepsize_scheduler.get_stepsize(t, psi_t)
        policy_update(pi_t, psi_t, eta_t, settings["update_rule"]) 

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

