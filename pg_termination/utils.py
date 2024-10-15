import numpy as np

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

