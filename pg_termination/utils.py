import numpy as np
import numpy.linalg as la

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

def set_rounded_policy(greedy_pi, pi):
    """ 
    :param greedy_pi: policy that be updated in place to the greedy policy 
    :param psi: advantage function
    :param states: index of states (to prevent memory reallocation)
    """
    greedy_pi[:,:] = 0
    (_, n_states) = pi.shape
    greedy_as = np.argmin(pi, axis=0)
    greedy_pi[greedy_as, np.arange(n_states)] = 1.

def kl_policy_update(psi, pi, eta, pi_scratch):
    (n_states, n_actions) = psi.shape

    if np.any(np.isnan(psi)):
        return False
    mult = np.exp(-eta*(psi - np.outer(np.min(psi, axis=1), np.ones(n_actions)))).T
    # # TODO: Why does adding this improve performance so much (for SPMD+CTD)?
    # if np.any(pi * mult <= 1e-32):
    #     return
    pi_scratch[:,:] = pi
    pi_scratch *= mult
    pi_sum = np.sum(pi_scratch, axis=0)
    if np.any(pi_sum == 0):
        return False

    pi[:,:] = pi_scratch
    pi /= np.outer(np.ones(n_actions), pi_sum)
    return True

def tsallis_policy_update(psi, pi, eta, gamma, pi_scratch):
    """ Policy update for Tsallis Inf. Solves up to accuracy 1e-6 if warm-start
    and 1e-8 for cold start.

    We want to solve
    $$
        min_{x in Delta_n} { <psi + eta^{-1}*u**(p-1)/(1-p), x> + eta^{-1} Psi(x) }
    $$
    where $u$ is the most recent policy and
    $$
        Psi(x) := sumlimits_{i=1}^n -(x_i^p)/[(1-p)p]
    $$
    Let us denote g = psi + eta^{-1}*u**(p-1)/(1-p). 

    By KKT conditions, we want to find ({x_i}_i, y=lam) such 
    $$
        g_i - eta^{-1}/(1-p) * x_i^(p-1) + y = 0, for all i
    $$
    and x is probability simplex. Solving for x,
    $$
        x_i = [ (1-p) * eta * (g_i + y) ]^(1/(p-1))
    $$
    So we compute this x_i and do binary search on y until this quantity x_i is a probability simplex

    :param grad: current gradient estimate
    :param pi: current policy
    :param eta: step size
    :param lam: lam to use for warm start
    """
    (n_S, n_A) = psi.shape
    assert n_A == pi.shape[0] and n_S == pi.shape[1], "psi (shape [%d, %d]) and pi (shape [%d, %d]) are not transpose to each other" % (n_S, n_A, pi.shape[0], pi.shape[1])

    tol = 1e-6
    p = 1/(1 - np.log(1.-gamma)/np.log(2))

    # update state-by-state separately
    for s in range(n_S):
        g = psi[s,:] + np.power(pi[:,s], p-1)/((1.-p) * eta)
        get_x_star = lambda y : np.power(np.maximum(tol, (1.-p) * eta * (g + y)), 1./(p-1))
        lam = tsallis_update_lambda(get_x_star, tol)
        pi_scratch[:,s] = get_x_star(lam)
        pi_sum_s = np.sum(pi_scratch[:,s])
        if pi_sum_s == 0:
            return False
        pi_scratch[:,s] /= pi_sum_s

    pi[:,:] = pi_scratch
    return True

def tsallis_update_lambda(get_x_star, tol):
    # TODO: Add back Newton warm-start?

    # cold start (if no warm_start flag or warm_start failed after 12 iterations)
    direction = 0

    # find whether lam should be positive or negative
    u = get_x_star(0)
    if abs(np.sum(u) - 1) <= tol:
        x_star = u/np.sum(u)
        return x_star
    elif np.sum(u) > 1:
        direction = 1
    else:
        direction = -1
        
    # exponential search
    lam = direction
    no_exp_search = True
    u = get_x_star(lam)
    while (direction == 1 and np.sum(u) > 1) or (direction == -1 and np.sum(u) < 1):
        no_exp_search = False
        lam *= 2
        u = get_x_star(lam)

    # binary search 
    if direction == 1:
        lo = 0 if no_exp_search else lam/(2.) 
        hi = lam # value that made sum(u) <= 1
        while hi-lo > tol:
            lam = (hi+lo)/2
            u = get_x_star(lam)
            if abs(np.sum(u) - 1)<=tol: break
            elif np.sum(u) < 1: hi = lam # makes the sum larger
            else: lo = lam
        best_lam = hi
    else:
        lo = lam # value that made sum(u) >= 1
        hi = 0 if no_exp_search else lam/(2.)
        while hi-lo > tol:
            lam = (hi+lo)/2
            u = get_x_star(lam)
            if abs(np.sum(u) - 1)<=tol: break
            elif np.sum(u) < 1: lo = lam
            else: hi = lam
        best_lam = lo

    return best_lam

def rand_l2(A, rng, max_iters=100, tol=1e-1):
    """
    Compute approximate largest singular value via randomized initial point.

    Source: https://arxiv.org/pdf/2402.17873 (Section 3.1)
    """
    x_t = rng.normal(size=A.shape[1])
    xi_t_prev = 0

    for t in range(max_iters):
        q_t = x_t/la.norm(x_t, ord=2)
        x_t = A.T.dot(A.dot(q_t))
        xi_t = np.dot(q_t, x_t) # eig(A'A)
        zeta_t = xi_t**0.5 # sig(A) = sqrt(eig(A'A))
        if t > 0 and (abs(xi_t - xi_t_prev)/xi_t <= 1e-6):
            break   
        xi_t_prev = xi_t
    
    return zeta_t
