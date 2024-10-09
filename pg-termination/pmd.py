""" Basic PMD """
import numpy as np

from wbmdp import GridWorldWithTraps

TOL = 1e-10

SUBLINEAR = 0
LINEAR_GEOMETRIC = 1
LINEAR_ADAPTIVE = 2

EUCLIDEAN_UPDATE = 100
KL_UPDATE = 101

def set_greedy_policy(greedy_pi, psi, states):
    """
    :param greedy_pi: policy that will hold the greedy policy 
    :param psi: advantage function
    :param states: index of states (to prevent memory reallocation)
    """
    greedy_pi[:,:] = 0
    (n_states,_) = psi.shape
    greedy_as = np.argmin(psi, axis=1)
    greedy_pi[greedy_as, states] = 1.

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
    (n_states, n_actions) = psi.shape

    if update_rule == EUCLIDEAN_UPDATE:
        pi_gd = pi - eta*psi.T
        pi[:] = parallel_simplex_projection(pi_gd)
        """
        for s in range(n_states):
            pi[:,s] = simplex_projection(pi_gd[:,s])
        """
    elif update_rule == KL_UPDATE:
        for s in range(n_states):
            pi[:,s] *= np.exp(-eta*(psi[s,:] - np.min(psi[s,:])))
            pi[:,s] /= np.sum(pi[:,s])

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

    def get_stepsize(self, t, psi):
        if self.stepsize_rule == SUBLINEAR:
            return 1
        elif self.stepsize_rule == LINEAR_GEOMETRIC:
            return (1./self.gamma)**t
        elif self.stepsize_rule == LINEAR_ADAPTIVE:
            # dynamically update
            if t % (self.N * self.T) == 0:
                self.Delta = np.max(-psi)/(1.-self.gamma)
            eta = 2**(1+np.floor(t/self.N) % self.T)/self.Delta
            return eta

def main():
    args = dict({
        "stepsize_rule": LINEAR_ADAPTIVE,
        "update_rule": EUCLIDEAN_UPDATE,
        "n_iters": 5000,
    })
    env = GridWorldWithTraps(10, 10, 0.99)

    pi_0 = np.ones((env.n_actions, env.n_states), dtype=float)/env.n_actions
    pi_t = pi_0
    greedy_pi_t = np.copy(pi_0)
    states = np.arange(env.n_states)

    stepsize_scheduler = StepsizeSchedule(env, args["stepsize_rule"])

    for t in range(args["n_iters"]):
        (psi_t, V_t) = env.get_advantage(pi_t)

        # check termination
        if np.max(-psi_t) < TOL:
            print("Iter %d: f=%.2e (gap=%.2e) (ggap=%.2e)" % (t+1, np.mean(V_t), np.max(-psi_t), np.max(-greedy_psi_t)))
            exit(0)

        # check termination of greedy
        set_greedy_policy(greedy_pi_t, psi_t, states)
        (greedy_psi_t, greedy_V_t) = env.get_advantage(greedy_pi_t)

        if (t+1) % 100 == 0:
            print("Iter %d: f=%.2e (gap=%.2e) (ggap=%.2e)" % (t+1, np.mean(V_t), np.max(-psi_t), np.max(-greedy_psi_t)))

        if np.max(-greedy_psi_t) < TOL:
            print("Iter %d: f=%.2e (gap=%.2e) (ggap=%.2e)" % (t+1, np.mean(V_t), np.max(-psi_t), np.max(-greedy_psi_t)))
            exit(0)

        eta_t = stepsize_scheduler.get_stepsize(t, psi_t)
        
        policy_update(pi_t, psi_t, eta_t, args["update_rule"]) 

if __name__ == '__main__':
    main()

