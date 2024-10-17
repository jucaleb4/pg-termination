""" White box MDPs (wbmdp) defined five-tuple M=(S,A,c,P,gamma) """

import numpy as np
import numpy.linalg as la

TOL = 1e-10

# Right (0), Down (1), Left (2), Up (3)
DIRS = [(1,0), (0,1), (-1,0), (0,-1)]

class MDPModel():
    """ Base MDP class """
    def __init__(self, n_states, n_actions, c, P, gamma, seed=None):
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
        assert np.max(np.sum(P, axis=0)) <= 1+TOL, \
            "P is not stochastic, recieved a sum of %.2f at (s,a)=(%d,%d)" % ( \
                np.max(np.sum(P, axis=0)), \
                np.where(1+TOL < np.sum(P, axis=0))[0][0], \
                np.where(1+TOL < np.sum(P, axis=0))[1][0], \
            )

        self.n_states = n_states
        self.n_actions = n_actions
        self.c = c
        self.P = P
        self.gamma = gamma

        # initialize a 
        self.rng = np.random.default_rng(seed)
        self.s = self.rng.integers(0, self.n_states)

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

    def estimate_advantage(self, pi, T, threshold=0):
        """
        https://arxiv.org/pdf/2303.04386

        :param T: duration to run Monte Carlo simulation
        :param threshold: pi(a|s) < threshold means Q(s,a)=largest value, do not visit again (rec: (1-gamma)**2/|A|)
        :return visited_state_action: whether a state-action pair was visited
        """
        costs = np.zeros(T, dtype=float)
        states = np.zeros(T, dtype=int)
        actions = np.zeros(T, dtype=int)

        for t in range(T):
            states[t] = self.s
            actions[t] = a_t = self.rng.choice(pi.shape[0], p=pi[:,states[t]])
            self.s = self.rng.choice(self.P.shape[0], p=self.P[:,states[t],actions[t]])

            costs[t] = self.c[states[t], actions[t]]

        cumulative_discounted_costs = np.zeros(T, dtype=float)
        cumulative_discounted_costs[-1] = costs[-1]
        for t in range(T-2,-1,-1):
            cumulative_discounted_costs[t] = costs[t] + self.gamma*cumulative_discounted_costs[t+1]

        # form advantage (dp style)
        Q = np.zeros((self.n_states, self.n_actions), dtype=float)
        visited_state_action = np.zeros((self.n_states, self.n_actions), dtype=bool)
        for t in range(T):
            (s,a) = states[t], actions[t]
            if visited_state_action[s,a]:
                continue
            Q[s,a] = cumulative_discounted_costs[t]

        # for proabibilities that are very low, set Q value to be high
        (poor_sa_a, poor_sa_s) = np.where(pi <= threshold)
        Q_max = np.max(np.abs(self.c))/(1.-self.gamma)
        Q[poor_sa_s,poor_sa_a] = Q_max

        V_pi = np.einsum('sa,as->s', Q, pi)
        psi = Q - np.outer(V_pi, np.ones(self.n_actions, dtype=float))

        return (psi, V_pi, visited_state_action)

    def get_steadystate(self, pi):
        P_pi = np.einsum('psa,as->ps', self.P, pi)

        dim = P_pi.shape[0]
        Q = (P_pi.T-np.eye(dim))
        ones = np.ones(dim)
        Q = np.c_[Q,ones]
        QTQ = np.dot(Q, Q.T)

        # check singular
        if la.matrix_rank(QTQ) < QTQ.shape[0]:
            print("Singular matrix when computing stationary distribution, return zero vector")
            return np.zeros(QTQ.shape[0], dtype=float)

        bQT = np.ones(dim)
        return np.linalg.solve(QTQ,bQT)

class GridWorldWithTraps(MDPModel):

    def __init__(self, length, n_traps, gamma, eps=0.05, seed=None, ergodic=False):
        """ Creates 2D gridworld with side length @length grid world with traps.

        Each step incurs a cost of +1
        @n_traps traps are randomly placed. Stepping on it will incur a high an addition cost of +5
        Reaching the target state will incur a cost of +0 and the agent will remain there.

        If :ergodic:=True mode, then reaching the target incurs a -length cost
        and the next state is a random non-target non-trap state. This ensures
        all state-action spaces can be visited after reaching the target.

        The agent can move in one of the four cardinal directions, if feasible. 
        There is a @eps probability another random direction will be selected.
        """

        n_states = length*length
        n_actions = 4
        n_traps = min(n_traps, n_states-1)

        rng = np.random.default_rng(seed)
        rnd_pts = rng.choice(length*length, replace=False, size=n_traps+1)
        traps = rnd_pts[:-1]
        self.target = target = rnd_pts[-1]
        print("Target at index %d" % target)

        P = np.zeros((n_states, n_states, n_actions), dtype=float)
        c = np.zeros((n_states, n_actions), dtype=float)

        def fill_gw_P_at_xy(P, x, y):
            """ 
            Applies standard probability in the 4 cardinal directions provided by @x and @y 

            :param x: x-axis locations of source we want to move from
            :param y: y-axis locations of source we want to move from
            :param length: length of x and y-axis
            :param eps: random probability of moving in another direction
            """
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
            fill_gw_P_at_xy(P, x, y)

        # vertical edges
        for i in range(2):
            x = (length-1)*i
            y = np.arange(1,length-1)
            fill_gw_P_at_xy(P, x, y)

        # horizontal edges
        for i in range(2):
            y = (length-1)*i
            x = np.arange(1,length-1)
            fill_gw_P_at_xy(P, x, y)

        # inner squares
        x = np.kron(np.ones(length, dtype=int), np.arange(1, length-1))
        y = np.kron(np.arange(1, length-1), np.ones(length, dtype=int))
        fill_gw_P_at_xy(P, x, y)

        # target
        if ergodic:
            rnd_pts = rng.choice(length*length, replace=False, size=n_traps+1)
            non_target_nor_trap = np.setdiff1d(np.arange(length*length), rnd_pts)

            P[:,target,:] = 0
            # go to random non-target non-trap location
            P[non_target_nor_trap,target,:] = 1./len(non_target_nor_trap)
        else:
            P[:,target,:] = 0
            # stay at target
            P[target,target,:] = 1.

        # apply trap cost
        c[:,:] = 1.
        c[traps,:] = 5.
        c[target,:] = 0

        super().__init__(n_states, n_actions, c, P, gamma)

    def get_target(self):
        return self.target

class GridWorldWithTrapsAndHills(MDPModel):

    def __init__(self, length, n_traps, gamma, eps=0.05, seed=None, ergodic=False):
        """ Same 2D gridworld, but the probablity of moving towards the target
        gets harder as you get closer.

        Let the current location b (x,y) and the target location (t_x,t_y). To move 
        to (x',y'). If either |t_x-x'| < |t_x-x| or |t_y-y'| < |t_y-y|, then
        the probability of successively moving in that direciton is
        (1-eps)*1/(length-min(|t_x-x'|, |t_y-y'|)). If not successful, we stay
        at the same location.
        """

        n_states = length*length
        n_actions = 4
        n_traps = min(n_traps, n_states-1)

        rng = np.random.default_rng(seed)
        rnd_pts = rng.choice(length*length, replace=False, size=n_traps+1)
        traps = rnd_pts[:-1]
        self.target = target = rnd_pts[-1]
        target_x = target % length
        target_y = target // length
        print("Target at index %d" % target)

        P = np.zeros((n_states, n_states, n_actions), dtype=float)
        c = np.zeros((n_states, n_actions), dtype=float)

        def fill_gw_P_at_xy(P, x, y):
            """ 
            Applies standard probability in the 4 cardinal directions provided by @x and @y 

            :param x: x-axis locations of source we want to move from
            :param y: y-axis locations of source we want to move from
            :param length: length of x and y-axis
            :param eps: random probability of moving in another direction
            """
            s = length*y+x
            for a in range(4):
                next_x = np.clip(x + DIRS[a][0], 0, length-1)
                next_y = np.clip(y + DIRS[a][1], 0, length-1)
                next_s = length*next_y+next_x

                next_x_is_closer = np.abs(next_x - target_x) < np.abs(x - target_x)
                next_y_is_closer = np.abs(next_y - target_y) < np.abs(y - target_y)

                next_s_proximity = 1 + np.minimum(np.abs(next_x - target_x), np.abs(next_y - target_y))
                next_s_is_closer = next_x_is_closer | next_y_is_closer

                P[next_s, s, a] = (1.-eps) * (eps*np.multiply(next_s_is_closer, 1./next_s_proximity) + (1.-next_s_is_closer))
                P[s, s, a] += (1.-eps) * np.multiply(next_s_is_closer, 1.-1./next_s_proximity) + (1.-eps)**2*np.multiply(next_s_is_closer, 1./next_s_proximity) 

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
            fill_gw_P_at_xy(P, x, y)

        # vertical edges
        for i in range(2):
            x = (length-1)*i
            y = np.arange(1,length-1)
            fill_gw_P_at_xy(P, x, y)

        # horizontal edges
        for i in range(2):
            y = (length-1)*i
            x = np.arange(1,length-1)
            fill_gw_P_at_xy(P, x, y)

        # inner squares
        x = np.kron(np.ones(length, dtype=int), np.arange(1, length-1))
        y = np.kron(np.arange(1, length-1), np.ones(length, dtype=int))
        fill_gw_P_at_xy(P, x, y)

        # target
        if ergodic:
            rnd_pts = rng.choice(length*length, replace=False, size=n_traps+1)
            non_target_nor_trap = np.setdiff1d(np.arange(length*length), rnd_pts)

            P[:,target,:] = 0
            # go to random non-target non-trap location
            P[non_target_nor_trap,target,:] = 1./len(non_target_nor_trap)
        else:
            P[:,target,:] = 0
            # stay at target
            P[target,target,:] = 1.

        # apply trap cost
        c[:,:] = 1.
        c[traps,:] = 5.
        c[target,:] = 0

        super().__init__(n_states, n_actions, c, P, gamma)

    def get_target(self):
        return self.target

class Taxi(MDPModel):

    # R, Y, G, B (x,y)
    color_arr = [(0,0), (0,4), (4,0), (3,4)]

    right_wall_arr = [(1,0), (1,1), (0,3), (0,4), (2,3), (2,4)]
    left_wall_arr  = [(2,0), (2,1), (1,3), (1,4), (3,3), (3,4)]

    def __init__(self, gamma, eps=0., ergodic=False):
        """ Creates 2D gridworld of fixed length=5 with a passenger at one of
        the 4 locations that needs to be dropped off at one of the hotel locations.
        The map appears as (see color_arr):

            +---------+
            |R: | : :G|
            | : | : : |
            | : : : : |
            | | : | : |
            |Y| : |B: |
            +---------+

        Based on: https://gymnasium.farama.org/environments/toy_text/taxi/

        Each step incurs a cost of +1.
        Correctly dropping off the passenger incurs a "cost" of -20.
        Illegally picking up or dropping a passenger incurs a high cost of 10.

        The agent can move in one of the four cardinal directions, if feasible. 
        There is a @eps probability another random direction will be selected.
        In addition, there are two additional actions: pickup and drop off.
        """
        length = 5

        # 5 locations for passenger (pass_loc=4 means it is in taxi), and 4 destinations
        n_states = length*length*5*4
        n_actions = 6

        P = np.zeros((n_states, n_states, n_actions), dtype=float)
        c = np.zeros((n_states, n_actions), dtype=float)

        def fill_gw_P_at_xy(P, x, y, length, eps):
            """ 
            Applies standard probability in the 4 cardinal directions provided by @x and @y 

            :param x: x-axis locations of source we want to move from
            :param y: y-axis locations of source we want to move from
            :param length: length of x and y-axis
            :param eps: random probability of moving in another direction
            """
            offsets = [p_loc*25+d_loc*125 for d_loc in range(4) for p_loc in range(5)]
            s = length*y+x
            for a in range(4):
                next_x = np.clip(x + DIRS[a][0], 0, length-1)
                next_y = np.clip(y + DIRS[a][1], 0, length-1)
                for offset in offsets:
                    curr_s = s + offset
                    next_s = length*next_y+next_x+offset
                    P[next_s, curr_s, a] = (1.-eps)

                # random action
                for b in range(4):
                    if b==a: continue
                    next_x = np.clip(x + DIRS[b][0], 0, length-1)
                    next_y = np.clip(y + DIRS[b][1], 0, length-1)
                    for offset in offsets:
                        curr_s = s + offset
                        next_s = length*next_y+next_x+offset
                        P[next_s, curr_s, a] += eps/3 # add to not over-write

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

        # hit a wall
        for right_wall in self.right_wall_arr:
            loc_x, loc_y = right_wall
            taxi_state = loc_x + 5*loc_y
            offsets = [p_loc*25+d_loc*125 for d_loc in range(4) for p_loc in range(5)]
            for offset in offsets:
                curr_state = taxi_state + offset
                # See DIRS
                P[:, curr_state, 0] = 0
                P[curr_state, curr_state, 0] = 1
            
        for left_wall in self.left_wall_arr:
            loc_x, loc_y = right_wall
            taxi_state = loc_x + 5*loc_y
            offsets = [p_loc*25+d_loc*125 for d_loc in range(4) for p_loc in range(5)]
            for offset in offsets:
                curr_state = taxi_state + offset
                # See DIRS
                P[:, curr_state, 2] = 0
                P[curr_state, curr_state, 2] = 1

        # apply step cost
        c[:,:] = 1.

        # (illegal) passenger pickup and drop off
        all_state_arr = np.arange(5*5*5*4)
        P[all_state_arr, all_state_arr, 4] = 1
        P[all_state_arr, all_state_arr, 5] = 1
        c[all_state_arr, 4] = 10
        c[all_state_arr, 5] = 10

        # legal passenger pickup
        for i, (x,y) in enumerate(self.color_arr):
            s = length*y+x
            old_passenger_loc = 25*i
            passenger_in_taxi_loc = 25*4
            destination_loc_arr = 125*np.arange(4)

            curr_state_arr = s + old_passenger_loc + destination_loc_arr
            next_state_arr = s + passenger_in_taxi_loc + destination_loc_arr

            P[:, curr_state_arr, 4] = 0
            P[next_state_arr, curr_state_arr, 4] = 1
            c[curr_state_arr, 4] = 1

        # we can only start where passenger is neither in taxi nor destination
        starting_states = np.array([], dtype=int)
        for passenger_loc in range(4):
            for destination_loc in range(4):
                if passenger_loc == destination_loc:
                    break
                offset = passenger_loc*25 + destination_loc*125
                starting_states = np.append(starting_states, np.arange(25)+offset)

        # legal passenger dropoff
        for i, (x,y) in enumerate(self.color_arr):
            s = length*y+x 
            old_passenger_loc = 25*4
            new_passenger_loc = 25*i
            destination_loc = 125*i

            curr_state_loc = s + old_passenger_loc + destination_loc
            next_state_loc = s + new_passenger_loc + destination_loc

            if ergodic:
                P[:, curr_state_loc, 5] = 0
                P[starting_states, curr_state_loc, 5] = 1./len(starting_states)
            else:
                P[:, curr_state_loc, 5] = 0
                P[next_state_loc, curr_state_loc, 5] = 1
                P[:, next_state_loc, :] = 0
                P[next_state_loc, next_state_loc, :] = 1
                c[next_state_arr, :] = 0

            c[curr_state_arr, 5] = -20

        super().__init__(n_states, n_actions, c, P, gamma)

class Random(MDPModel):
    def __init__(self, n_states, n_actions, gamma, seed=None):
        rng = np.random.default_rng(seed)
        P = rng.integers(0, int(1e6), size=(n_states, n_states, n_actions)).astype(float)
        sum_P = np.sum(P, axis=0)
        P /= sum_P
        c = rng.normal(size=(n_states, n_actions))

        super().__init__(n_states, n_actions, c, P, gamma)

class Small(MDPModel):
    def __init__(self, n_states, gamma, eps=1e-8, seed=None):
        rng = np.random.default_rng(seed)

        n_actions = n_states
        all_states_arr = np.arange(n_states)
        general_states_arr = np.arange(1, n_states)

        P = rng.uniform(size=(n_states, n_states, n_actions)).astype(float)
        # make it rare to go to island state from non-island state
        P[0, general_states_arr, :] = 1e-10

        # normalize probabilities
        sum_P = np.sum(P, axis=0)
        P /= sum_P
        c = rng.normal(scale=0.1, size=(n_states, n_actions))

        # general states going to island state
        for s in general_states_arr:
            P[0, s, 0] = eps
            u = rng.uniform(size=len(general_states_arr))
            u /= np.sum(u)
            # other distributions go elsewhere
            P[general_states_arr, s, 0] = (1.-eps)*u
            c[s, 0] = 10

        # island state staying to itself
        P[0, 0, 0] = 1-eps
        P[general_states_arr, 0, 0] = eps/(len(general_states_arr)-1.)
        c[0, :] = -(1./eps)

        sum_P = np.sum(P, axis=0)
        P /= sum_P

        super().__init__(n_states, n_actions, c, P, gamma)

class Chain(MDPModel):
    def __init__(self, n_states, gamma, eps=1e-8, seed=None):
        rng = np.random.default_rng(seed)

        n_actions = n_states

        P = np.zeros((n_states, n_states, n_actions), dtype=float)
        c = np.zeros((n_states, n_actions), dtype=float)
        all_states_arr = np.arange(n_states)

        # general states going to island state
        for s in all_states_arr:
            for a in range(n_actions):
                u = rng.uniform(low=-eps, high=eps)
                P[s, s, a] = (1.-eps)+u
                P[(s+1)%n_states, s, a] = eps-u
                c[s,a] = rng.normal()

        super().__init__(n_states, n_actions, c, P, gamma)

def get_env(name, gamma, seed=None):
    if name == "gridworld_small":
        env = GridWorldWithTraps(20, 20, gamma, seed=seed, ergodic=True)
    elif name == "gridworld_large":
        env = GridWorldWithTraps(50, 50, gamma, seed=seed, ergodic=True)
    elif name == "gridworld_hill_small":
        env = GridWorldWithTrapsAndHills(20, 20, gamma, seed=seed, ergodic=True)
    elif name == "gridworld_hill_large":
        env = GridWorldWithTrapsAndHills(50, 50, gamma, seed=seed, ergodic=True)
    elif name == "taxi":
        env = Taxi(gamma, ergodic=True)
    elif name == "random":
        env = Random(100, 100, gamma, seed=seed)
    elif name == "chain":
        env = Chain(100, gamma, eps=1e-3, seed=seed)
    else:
        raise Exception("Unknown env_name=%s" % name)

    return env
