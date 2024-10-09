""" White box MDPs (wbmdp) defined five-tuple M=(S,A,c,P,gamma) """

import numpy as np
import numpy.linalg as la

TOL = 1e-10

DIRS = [(1,0), (0,1), (-1,0), (0,-1)]

class MDPModel():
    """ Base MDP class """
    def __init__(self, n_states, n_actions, c, P, gamma):
        assert len(c.shape) == 2, "Input cost vector c must be a 2-D vector, recieved %d dimensions" % len(c.shape)
        assert len(P.shape) == 3, "Input cost vector c must be a 3-D tensor, recieved %d dimensions" % len(P.shape)

        assert c.shape[0] == n_states, "1st dimension of c must equal n_states=%d, was instead %d" % (n_states, c.shape[0])
        assert c.shape[1] == n_actions, "2nd dimension of c must equal n_actions=%d, was instead %d" % (n_actions, c.shape[1])
        assert P.shape[0] == n_states, "1st dimension of P must equal n_states=%d, was instead %d" % (n_states, P.shape[0])
        assert P.shape[1] == n_states, "2nd dimension of P must equal n_states=%d, was instead %d" % (n_states, P.shape[1])
        assert P.shape[2] == n_actions, "3rd dimension of P must equal n_actions=%d, was instead %d" % (n_actions, P.shape[2])
        assert 0 < gamma < 1, "Input discount gamma must be (0,1), recieved %f" % gamma

        assert 1-TOL <= np.min(np.sum(P, axis=0)), \
            "P is not stochastic, recieved a sum of %.2f at (s,a)=(%d,%d)" % ( \
                np.min(np.sum(P, axis=0)), \
                np.where(1-TOL > np.sum(P, axis=0))[0][0], \
                np.where(1-TOL > np.sum(P, axis=0))[1][0], \
            )
        assert np.min(np.sum(P, axis=0)) <= 1+TOL, \
            "P is not stochastic, recieved a sum of %.2f at (s,a)=(%d,%d)" % ( \
                np.min(np.sum(P, axis=0)), \
                np.where(1+TOL < np.sum(P, axis=0))[0][0], \
                np.where(1+TOL < np.sum(P, axis=0))[1][0], \
            )

        self.n_states = n_states
        self.n_actions = n_actions
        self.c = c
        self.P = P
        self.gamma = gamma

    def get_advantage(self, pi):
        assert pi.shape[0] == self.n_actions, "1st dimension of pi must equal n_actions=%d, was instead %d" % (self.n_actions, pi.shape[0])
        assert pi.shape[1] == self.n_states, "2nd dimension of pi must equal n_states=%d, was instead %d" % (self.n_states, pi.shape[1])

        # sum over actions (p=s' next state, s curr state, a action)
        P_pi = np.einsum('psa,as->ps', self.P, pi)
        c_pi = np.einsum('sa,as->s', self.c, pi)

        # (I-gamma*(P^pi)')V = c^pi
        V_pi = la.solve(np.eye(self.n_states) - self.gamma*P_pi.T, c_pi)
        Q_pi = self.c + self.gamma*np.einsum('psa,p->sa', self.P, V_pi)
        psi = Q_pi - np.outer(V_pi, np.ones(self.n_actions))

        return (psi, V_pi)

class GridWorldWithTraps(MDPModel):

    def __init__(self, length, n_traps, gamma, eps=0.05, seed=None):
        """ Creates 2D gridworld with side length @length grid world with traps.

        Each step incurs a cost of +1
        @n_traps traps are randomly placed. Stepping on it will incur a high an addition cost of +5
        Reaching the target state will incur a cost of +0 and the agent will remain there.

        The agent can move in one of the four cardinal directions, if feasible. 
        There is a @eps probability another random direction will be selected.
        """

        n_states = length*length
        n_actions = 4
        n_traps = min(n_traps, n_states-1)

        rng = np.random.default_rng(seed)
        rnd_pts = rng.choice(length*length, replace=False, size=n_traps+1)
        traps = rnd_pts[:-1]
        target = rnd_pts[-1]

        P = np.zeros((n_states, n_states, n_actions), dtype=float)
        c = np.zeros((n_states, n_actions), dtype=float)

        def fill_gw_P_at_xy(P, x, y, length, eps):
            s = length*y+x
            for a in range(4):
                next_x = np.clip(x + DIRS[a][0], 0, length-1)
                next_y = np.clip(y + DIRS[a][1], 0, length-1)
                next_s = length*next_y+next_x
                P[next_s, s, a] = (1.-eps)

                # random action
                for b in range(4):
                    if b==a: continue
                    next_x = np.clip(x + DIRS[b][0], 0, length-1)
                    next_y = np.clip(y + DIRS[b][1], 0, length-1)
                    next_s = length*next_y+next_x
                    P[next_s, s, a] += eps/3 # add to not over-write

        # handle corners
        for i in range(4):
            x = (length-1)*(i%2)
            y = (length-1)*(i//2)
            fill_gw_P_at_xy(P, x, y, length, eps)

        # vertical edges
        for i in range(2):
            x = (length-1)*i
            y = np.arange(1,length-1)
            fill_gw_P_at_xy(P, x, y, length, eps)

        # horizontal edges
        for i in range(2):
            y = (length-1)*i
            x = np.arange(1,length-1)
            fill_gw_P_at_xy(P, x, y, length, eps)

        # inner squares
        x = np.kron(np.ones(length, dtype=int), np.arange(1, length-1))
        y = np.kron(np.arange(1, length-1), np.ones(length, dtype=int))
        fill_gw_P_at_xy(P, x, y, length, eps)

        # target
        P[:,target,:] = 0
        P[target,target,:] = 1.

        # apply trap cost
        c[:,:] = 1.
        c[traps,:] = 5.
        c[target,:] = 0.

        super().__init__(n_states, n_actions, c, P, gamma)
