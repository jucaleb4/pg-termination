""" Basic PMD """
import time
import os

import numpy as np

from wbmdp import GridWorldWithTraps
from logger import BasicLogger

TOL = 1e-10

SUBLINEAR = 0
KL_LINEAR_GEOMETRIC = 1
KL_LINEAR_ADAPTIVE = 2
EUCLIDEAN_LINEAR_ADAPTIVE = 3

EUCLIDEAN_UPDATE = 100
KL_UPDATE = 101

def set_greedy_policy(greedy_pi, psi):
    """ 
    :param greedy_pi: policy that be updated in place to the greedy policy 
    :param psi: advantage function
    :param states: index of states (to prevent memory reallocation)
    """
    greedy_pi[:,:] = 0
    (n_states,_) = psi.shape
    greedy_as = np.argmin(psi, axis=1)
    greedy_pi[greedy_as, np.arange(n_states)] = 1.

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

    assert np.max(np.abs(np.sum(X_proj, axis=0) - 1)) <= TOL, "Sum of sum(x)=%.4e != 1.0" % np.max(np.sum(x_proj, axis=0))

    return X_proj

def policy_update(pi, psi, eta, update_rule):
    """ Projection onto simplex with Euclidean or closed-form solution with KL """
    (n_states, n_actions) = psi.shape

    if update_rule == EUCLIDEAN_UPDATE:
        pi_gd = pi - eta*psi.T
        pi[:] = parallel_simplex_projection(pi_gd)
    elif update_rule == KL_UPDATE:
        pi *= np.exp(-eta*(psi - np.outer(np.min(psi, axis=1), np.ones(n_actions)))).T
        pi /= np.outer(np.ones(n_actions), np.sum(pi, axis=0))

class StepsizeSchedule():
    """ Returns steps size """
    def __init__(self, env, stepsize_rule):
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.gamma = env.gamma
        self.stepsize_rule = stepsize_rule

        self.N = np.ceil(4./(1-self.gamma))
        self.T = np.ceil(np.log(self.n_states**3*self.n_actions/(1.-self.gamma)**2/np.log(2)))
        self.Delta = 0

        if stepsize_rule in [KL_UPDATE, EUCLIDEAN_LINEAR_ADAPTIVE]:
            print("Worse case iteration complexity: %d" % (self.n_states*(self.n_actions-1)*self.N*self.T))

    def get_stepsize(self, t, psi):
        if self.stepsize_rule == SUBLINEAR:
            return 1
        elif self.stepsize_rule == KL_LINEAR_GEOMETRIC:
            return (1./self.gamma)**t
        elif self.stepsize_rule == KL_LINEAR_ADAPTIVE:
            (_, n_actions) = psi.shape
            # dynamically update
            if t % (self.N * self.T) == 0:
                self.Delta = np.max(-psi)/(1.-self.gamma)
            eta = np.log(n_actions)*4**(np.floor(t/self.N) % self.T)*2**(self.T*np.floor(t/(self.T*self.N)))/self.Delta
            return eta
        elif self.stepsize_rule == EUCLIDEAN_LINEAR_ADAPTIVE:
            # dynamically update
            if t % (self.N * self.T) == 0:
                self.Delta = np.max(-psi)/(1.-self.gamma)
            eta = 2**(1+np.floor(t/self.N) % self.T)/self.Delta
            return eta

def main():
    args = dict({
        # "stepsize_rule": KL_LINEAR_GEOMETRIC,
        # "stepsize_rule": KL_LINEAR_ADAPTIVE, 
        "stepsize_rule": EUCLIDEAN_LINEAR_ADAPTIVE,
        # "update_rule": KL_UPDATE, 
        "update_rule": EUCLIDEAN_UPDATE,
        "n_iters": 10000,
        "fname": os.path.join("logs", "gridworld.csv")
    })
    env = GridWorldWithTraps(10, 10, 0.99, seed=1104)
    logger = BasicLogger(
        fname=args["fname"], 
        keys=["iter", "average value", "advantage gap", "greedy advantage gap"], 
        dtypes=['d', 'f', 'f', 'f']
    )

    pi_0 = np.ones((env.n_actions, env.n_states), dtype=float)/env.n_actions
    pi_t = pi_0
    greedy_pi_t = np.copy(pi_0)

    stepsize_scheduler = StepsizeSchedule(env, args["stepsize_rule"])

    s_time = time.time()
    # tolerance for optimality (due to floating point error)
    eps_tol = 1e-12 

    for t in range(args["n_iters"]):
        (psi_t, V_t) = env.get_advantage(pi_t)


        # check termination of greedy
        set_greedy_policy(greedy_pi_t, psi_t)
        (greedy_psi_t, greedy_V_t) = env.get_advantage(greedy_pi_t)

        if t <= 9 or (t <= 99 and (t+1) % 5 == 0) or (t+1) % 100 == 0:
            print("Iter %d: f=%.2e (gap=%.2e) (ggap=%.2e)" % (t+1, np.mean(V_t), np.max(-psi_t), np.max(-greedy_psi_t)))
            logger.log(t+1, np.mean(V_t), np.max(-psi_t), np.max(-greedy_psi_t))

        if np.max(-psi_t) < eps_tol or np.max(-greedy_psi_t) < eps_tol:
            print("Terminate at %d: f=%.2e (gap=%.2e) (ggap=%.2e)" % (t+1, np.mean(V_t), np.max(-psi_t), np.max(-greedy_psi_t)))
            logger.log(t+1, np.mean(V_t), np.max(-psi_t), np.max(-greedy_psi_t))
            break

        eta_t = stepsize_scheduler.get_stepsize(t, psi_t)
        
        policy_update(pi_t, psi_t, eta_t, args["update_rule"]) 

    print("Total runtime: %.2fs" % (time.time() - s_time))

    logger.save()

if __name__ == '__main__':
    main()
