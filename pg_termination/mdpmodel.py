""" Models for MDPs defined five-tuple M=(S,A,c,P,gamma) """

import numpy as np
import numpy.linalg as la
import abc
import time

import gymnasium as gym

TOL = 1e-10

# Right (0), Down (1), Left (2), Up (3)
DIRS = [(1,0), (0,1), (-1,0), (0,-1)]

class MDPModel():
    """ 
    Base MDP class. 

    Online estimation schemes are also implemented here since they hold for all
    models we consider (known/generative model and online model).

    We use Gymnasium's template (https://gymnasium.farama.org/api/env/) to
    require `step(a)`, where `a` is the input action

    """
    def __init__(self, gamma, seed=None):
        assert 0 < gamma < 1, "Input discount gamma must be (0,1), recieved %f" % gamma
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma

    @abc.abstractmethod
    def step(self, a):
        """ 
        Takes a single time step of the MDP.

        :param a: action (must be an int)
        :return s: state
        :return c: cost
        :return terminate: termination/reset
        """
        return

    @abc.abstractmethod
    def get_mixing_time_ub(self, pi):
        """ Computes exact mixing under pi. If unknown, returns arbitrary.
        :returns t_mix: mixing time 
        :returns nu: stationary distribution
        """
        return

    def estimate_advantage_online_mc(self, pi, T, threshold=0, time_limit=np.inf):
        """
        https://arxiv.org/pdf/2303.04386

        :param T: duration to run Monte Carlo simulation
        :param threshold: pi(a|s) < threshold means Q(s,a)=largest value, do not visit again (rec: (1-gamma)**2/|A|)
        :return visit_len_state_action: how long the Monte carlo estimate is at every state-aciton pair
        """
        assert T >= 1, "Mixing time T=%d smaller than 1" % T

        init_size = 1024
        costs = np.zeros(init_size, dtype=float)
        states = np.zeros(init_size, dtype=int)
        actions = np.zeros(init_size, dtype=int)
        s_time = time.time()
        early_terminate = False
        psi = np.zeros((self.n_states, self.n_actions), dtype=float)
        V_pi = np.zeros(self.n_states, dtype=float)

        has_adjusted_time = False
        T_time_adjusted = np.inf
        # run Monte Carlo while estimating 1 percentage of total runtime
        for t in range(T):
            states[t] = self.s
            actions[t] = self.rng.choice(pi.shape[0], p=pi[:,states[t]])
            (_, costs[t], _) = self.step(actions[t])

            # early termination due to time
            if (not has_adjusted_time) and ((time.time() - s_time) >= 0.01 * time_limit):
                has_adjusted_time = True
                # increase 67X due acct for Q (assume sampling is ~2/3 of time)
                T_time_adjusted = int(min(T, 55*(t+1)))
            elif t > T_time_adjusted:
                early_terminate = True
                return (early_terminate, psi, V_pi, t)
            elif (t+1) == len(costs):
                costs = np.append(costs, np.zeros(len(costs)))
                states = np.append(states, np.zeros(len(states), dtype=int))
                actions = np.append(actions, np.zeros(len(actions), dtype=int))

        costs = costs[:T]
        states = states[:T]
        actions = actions[:T]

        # form advantage (dp style); 
        cumulative_discounted_costs = np.zeros(T, dtype=float)
        cumulative_discounted_costs[-1] = costs[-1]
        for t in range(T-2,-1,-1):
            cumulative_discounted_costs[t] = costs[t] + self.gamma*cumulative_discounted_costs[t+1]

        Q = np.zeros((self.n_states, self.n_actions), dtype=float)
        visit_len_state_action = np.zeros((self.n_states, self.n_actions), dtype=bool)
        for t in range(T):
            (s,a) = states[t], actions[t]
            if visit_len_state_action[s,a] > 0:
                continue
            Q[s,a] = cumulative_discounted_costs[t]
            visit_len_state_action[s,a] = T-t

        # for proabibilities that are very low, set Q value to be high
        (poor_sa_a, poor_sa_s) = np.where(pi <= threshold)
        Q_max = np.max(np.abs(self.c))/(1.-self.gamma)
        Q[poor_sa_s,poor_sa_a] = Q_max

        V_pi = np.einsum('sa,as->s', Q, pi)
        psi = Q - np.outer(V_pi, np.ones(self.n_actions, dtype=float))

        return (early_terminate, psi, V_pi, T)

    def estimate_mixing_properties(self, pi, T, tmix=0, nu=None, time_limit=np.inf):
        """
        https://proceedings.neurips.cc/paper/2015/file/7ce3284b743aefde80ffd9aec500e085-Paper.pdf

        Note that this paper is for ergodic and reversible Markov chains. 
        Ergodic can be found: https://proceedings.mlr.press/v99/wolfer19a/wolfer19a.pdf

        However, the non-reversible is quite complicated, so we approximate with reversible for
        simplicity. We also avoid confidence intervals to simplify the implementations.

        :param T: number of samples to estimate mixing time. Overwritten if tmix and nu provided
        :param tmix: true tmix 
        :param nu: true stationary dist.
        :param time_limit: time limit for estimating mixing time
        :return nu_est: estimated stationary distribution
        :return tmix_est: estimated mixing time
        :return n_samples: samples used for estimation
        """
        nu_est = np.zeros(self.n_states, dtype=float)
        M_est = np.zeros((self.n_states, self.n_states), dtype=float)

        if (tmix > 0) and (nu is not None):
            trelax = tmix/np.log(2)
            spec_gap = 1./trelax
            nu_min = np.min(nu)
            # https://arxiv.org/pdf/2303.04386 (based on Thm 4.1)
            T = int(np.log(self.n_states)/(nu_min*spec_gap)+1)

        s_time = time.time()
        T_time_adjusted = np.inf # adjust later after time trial
        has_adjusted_time = False
        early_terminate = False

        curr_s = self.s
        for t in range(T):
            nu_est[curr_s] += 1
            a = self.rng.choice(pi.shape[0], p=pi[:,curr_s])
            self.step(a)
            M_est[curr_s,self.s] += 1
            curr_s = self.s

            # early termination due to time
            if (not has_adjusted_time) and ((time.time() - s_time) >= 0.01 * time_limit):
                has_adjusted_time = True
                T_time_adjusted = int(min(T, 110*(t+1)))
            elif t > T_time_adjusted:
                early_terminate = True
                return (early_terminate, 0, 0, t)

        # normalize and compute intermediates
        if np.min(nu_est) == 0:
            nu_est += 1e-6 # small perturbation
        nu_est /= T; M_est /= (T-1)
        nu_est_invsq = np.reciprocal(np.sqrt(nu_est))
        L = np.diag(nu_est_invsq)@M_est@np.diag(nu_est_invsq)
        sym_L = 0.5*(L + L.T)
        eig_sym_L = np.sort(la.eig(sym_L)[0].real)[::-1]

        # estimate spectral gap
        # if we have incorrect value, use heuristics
        nu_est_lb = np.min(nu_est) 
        spec_gap_est = 1 - max(eig_sym_L[1], abs(eig_sym_L[-1]))
        if (not (0 < spec_gap_est < 1)):
            eig_L = np.sort(la.eig(L)[0].real)[::-1]
            spec_gap_est = 1 - max(eig_L[1], abs(eig_L[-1]))
        if (not (0 < spec_gap_est < 1)):
            spec_gap_est = 1e-3
        trelax_est = 1./spec_gap_est
        tmix_est = trelax_est * np.log(4/nu_est_lb)

        n_samples = T
        return (early_terminate, nu_est, tmix_est, n_samples)

    def estimate_advantage_online_mc_dynamic(self, pi, eps, threshold=0, time_limit=np.inf):
        """

        :param eps: accuracy threshold
        :param threshold: pi(a|s) < threshold means Q(s,a)=largest value, do not visit again (rec: (1-gamma)**2/|A|)
        :return visit_len_state_action: how long the Monte carlo estimate is at every state-aciton pair
        """
        cycle = 1024
        T = cycle
        costs = np.zeros(T, dtype=float)
        states = np.zeros(T, dtype=int)
        actions = np.zeros(T, dtype=int)
        unvisited_sa = (pi >= threshold).astype(int)

        psi = np.zeros((self.n_states, self.n_actions), dtype=float)
        V_pi = np.zeros(self.n_states, dtype=float)

        t = 0
        s_time = time.time()
        has_adjusted_time = False
        T_time_adjusted = np.inf
        early_terminate = False

        countdown = max(1, np.log(1./eps))
        start_countdown = False
        while countdown > 0:
            states[t] = self.s
            actions[t] = self.rng.choice(pi.shape[0], p=pi[:,states[t]])
            (_, costs[t], _) = self.step(actions[t])
            unvisited_sa[actions[t], states[t]] = 0

            # check if we can terminate
            if t % cycle == 0:
                start_countdown = (np.max(unvisited_sa) == 0)
            if start_countdown:
                countdown -= 1

            # early termination due to time
            t += 1
            if t == len(costs):
                costs = np.append(costs, np.zeros(t))
                states = np.append(states, np.zeros(t, dtype=int))
                actions = np.append(actions, np.zeros(t, dtype=int))
            if (not has_adjusted_time) and ((time.time() - s_time) >= 0.01 * time_limit):
                has_adjusted_time = True
                T_time_adjusted = int(55*t)
            elif t > T_time_adjusted:
                early_terminate = True
                return (early_terminate, psi, V_pi, t)

        # form advantage (DP style)
        T = t # dynamic mixing time
        cumulative_discounted_costs = np.zeros(T, dtype=float)
        cumulative_discounted_costs[-1] = costs[-1]
        for t in range(T-2,-1,-1):
            cumulative_discounted_costs[t] = costs[t] + self.gamma*cumulative_discounted_costs[t+1]

        Q = np.max(np.abs(self.c))/(1.-self.gamma) * np.ones((self.n_states, self.n_actions), dtype=float)
        visit_len_state_action = np.zeros((self.n_states, self.n_actions), dtype=bool)
        for t in range(T):
            (s,a) = states[t], actions[t]
            if visit_len_state_action[s,a] > 0:
                continue
            Q[s,a] = cumulative_discounted_costs[t]
            visit_len_state_action[s,a] = T-t

        # for low probability s-a, set a high value 
        (poor_sa_a, poor_sa_s) = np.where(pi <= threshold)
        Q_max = np.max(np.abs(self.c))/(1.-self.gamma)
        Q[poor_sa_s,poor_sa_a] = Q_max

        V_pi = np.einsum('sa,as->s', Q, pi)
        psi = Q - np.outer(V_pi, np.ones(self.n_actions, dtype=float))

        return (early_terminate, psi, V_pi, T)

    def estimate_random_reset_value(self, pi, n_replicates=30, time_limit=np.inf):
        """
        Estimates the quantity

        $$
            E_(s ~ rho)[ E [ c_0 + gamma*c_1 + gamma**2 * c_2 + ... | s_0=s] ],
        $$
        
        where $rho$ is the reset distribution. This requires the environment to
        be have a termination condition in finite time to ensure non-infinite
        runtime. A runtime can be supplied.

        :param pi: policy 
        :param n_replicates: how many Monte Carlo replicates to estimate
        :param time_limit: maximum runtime to run for
        :return V_est: estimate of randomly reset value function
        """
        s_time = time.time()

        # run until reset
        terminate = 0
        s = 0
        while (not terminate):
            a = self.rng.choice(pi.shape[0], p=pi[:,s])
            (s, _, terminate) = self.step(a)
            if time.time() - s_time > time_limit:
                break

        t = 0
        replic_id = 0
        curr_V = 0
        V_est = 0
        while replic_id < n_replicates:
            a = self.rng.choice(pi.shape[0], p=pi[:,s])
            (s, c, terminate) = self.step(a)
            curr_V += (self.gamma**t) * c
            t += 1
            if terminate:
                replic_id += 1
                alpha = 1./replic_id
                V_est = (1.-alpha) * V_est + alpha*curr_V
                t = curr_V = 0
            if time.time() - s_time > time_limit:
                break

        return V_est if replic_id > 0 else np.inf

    def estimate_random_reset_advantage(self, pi, n_replicates=30, time_limit=np.inf):
        """
        Similar to `estimate_random_reset_value`, we estimate the quantity

        $$
            E_(s ~ rho)[ E [ c_0 + gamma*c_1 + gamma**2 * c_2 + ... | s_0=s, a_0=a] ],
        $$
        
        where $rho$ is the reset distribution. 

        :param pi: policy 
        :param n_replicates: number of Monte Carlo estimates 
        :param time_limit: maximum runtime to run for
        :return V_est: estimate of randomly reset value function
        """
        s_time = time.time()

        # run until reset
        terminate = 0
        s = 0
        while (not terminate):
            a = self.rng.choice(pi.shape[0], p=pi[:,s])
            (s, _, terminate) = self.step(a)
            if time.time() - s_time > time_limit:
                break

        t = 0
        replic_id = 0
        curr_Q = np.zeros(self.n_actions, dtype=float)
        Q_est = np.zeros(self.n_actions, dtype=float)
        action_id = 0 # which action we are estimating
        reset_state_hit = np.zeros(self.n_states, dtype=float) 
        reset_state_hit[s] += 1

        while replic_id < n_replicates:
            a = self.rng.choice(pi.shape[0], p=pi[:,s])
            if t == 0:
                a = action_id
                
            (s, c, terminate) = self.step(a)
            curr_Q[action_id] += (self.gamma**t) * c
            t += 1
            if terminate:
                t = 0
                reset_state_hit[s] += 1
                action_id = (action_id + 1) % self.n_actions
                if action_id == 0:
                    replic_id += 1
                    alpha = 1./replic_id
                    Q_est = (1.-alpha) * Q_est + alpha*curr_Q
                    curr_Q[:] = 0
            if time.time() - s_time > time_limit:
                break

        # convert Q to advantage
        reset_state_avg = reset_state_hit/np.sum(reset_state_hit)
        V_est = np.dot(Q_est, np.einsum("as,s->a", pi, reset_state_avg))
        A_est = Q_est - V_est

        return A_est if replic_id > 0 else np.inf * np.ones(self.n_actions)

    def estimate_advantage_online_ctd(
            self, pi, Phi, Phi_max, Phi_min, ukappa, iota_mult, state_expl,
            is_finite_state, time_limit=np.inf, max_obs=np.inf, s_origin=None,
            burn_in=False, N_mult=1.0, uLam_mult=1.0,
    ):
        """
        Forms nearly unbiased TD estimator that is bounded w.h.p.

        :params pi: policy
        :params Phi: feature matrix
        :params Phi_max, Phi_min: estimate of largest and small sig-val of Phi
        :params ukappa: lower bound estimate of minimum discounted visit distribution
        :params iota_mult: multiplier for iota (prefer >= 1)
        :parmas state_expl: whether we want explicit state exploration
        :params is_finite_state: TODO - whether state space is finite
        :params time_limit: amount of time left (in sec)
        :params max_obs: number of samples left (in sec)
        :params s_origin: Origin state. If 'None' (default), will choose reset
        :params N_mult: CTD N multiplier
        :params uLam_mult: uLam multiplier
        :returns early_terminate: whether time ran out
        :returns hpsi: estimate of advantage function
        :returns hV: estimate of state value
        :returns cum_samples: total number of samples used
        """
        # parameter setup (for operator F)
        L = Phi_max**2
        C1 = (Phi.shape[0]**2)*L
        C2 = L

        # parameter setup (for operator F with unknown kappa)
        uLam = uLam_mult * ukappa * Phi_min**2
        umu = (1.-self.gamma)*uLam
        oTheta = 1./np.sqrt((1.-self.gamma)*umu)
        theta = np.zeros(Phi.shape[1])

        # parameter setup (algorithmic terms)
        ell_0 = max((L/umu)**2, C2/umu**2)
        um   = max(np.log(1./umu), np.log(C1))/np.log(1./self.gamma)
        sig = np.sqrt(C2)*oTheta
        R    = np.sqrt(np.max([oTheta**2, sig**2/umu**2, np.sqrt(C2)/umu]))
        B_1  = np.max([
                np.sqrt((1.-self.gamma)**2*ell_0)/oTheta, 
                ((1.-self.gamma)*sig/umu)**2, 
                ((1.-self.gamma)**4 * R**2 * C2)/(umu**2)
        ])
        N    = int(max(1, N_mult*max(B_1, ell_0))) 
        # only print CTD iterations first time
        if not hasattr(self, "first_ctd_call"):
            print("N=%d" % N)
            self.first_ctd_call = False

        eps_expl_a = (1.-self.gamma)*ukappa
        eps_expl_s = (1.-self.gamma)/(-np.log(1.-self.gamma))**2 if state_expl else 0.

        theta_t = np.zeros(Phi.shape[1])
        cum_samples = 0
        s_time = time.time()
        early_terminate = False

        # random s_origin 
        if s_origin == 'rand':
            pi_expl_s = (1.-eps_expl_s)*pi + (eps_expl_s/self.n_actions)
            s_visit_count = np.zeros(self.n_states)
            for t in range(10*self.n_states):
                a = self.rng.choice(pi.shape[0], p=pi_expl_s[:,self.s])
                (self.s, _, terminate) = self.step(a)
                s_visit_count[self.s] += 1
            s_origin = s_visit_count/(10*self.n_states)

        for t in range(N):
            time_left = time_limit - (time.time() - s_time)
            obs_left = max_obs - cum_samples
            iota = iota_mult/((ell_0 + t)*umu)
            (early_terminate, hF_t, n_samples) = self._get_hF_estimate(
                pi, Phi, theta, um, eps_expl_a, eps_expl_s, time_left, 
                obs_left, s_origin, burn_in
            )
            if early_terminate:
                empty_psi = np.zeros((self.n_states, self.n_actions))
                empty_V = np.zeros(self.n_states)
                return (early_terminate, empty_psi, empty_V, cum_samples)
            theta_t -= iota*hF_t
            cum_samples += n_samples

        (n_actions, n_states) = pi.shape
        hQ = np.reshape(Phi@theta_t, newshape=(n_states,n_actions), order='C')
        hV = np.einsum('sa,as->s', hQ, pi)
        hpsi = hQ - np.outer(hV, np.ones(self.n_actions))

        return (early_terminate, hpsi, hV, cum_samples)

    def _hF_waiting_period(self, pi, eps_expl_s, s_time, time_limit, max_obs, s_origin, cum_samples):
        terminate = False # for env
        early_terminate = False # for budget
        checkpoint = 128
        is_s_origin_random = isinstance(s_origin, np.ndarray)

        pi_expl_s = (1.-eps_expl_s)*pi + (eps_expl_s/self.n_actions)
        while 1:
            if (s_origin is None) and terminate:
                break
            if (not is_s_origin_random) and self.s == s_origin:
                break
            if is_s_origin_random and (self.rng.random() < s_origin[self.s]):
                break
            if cum_samples == checkpoint:
                if ((time.time() - s_time) > time_limit) or (cum_samples > max_obs):
                    early_terminate = True
                    return (early_terminate, cum_samples)
                checkpoint *= 2
            a = self.rng.choice(pi.shape[0], p=pi_expl_s[:,self.s])
            (self.s, _, terminate) = self.step(a)
            cum_samples += 1

        return (early_terminate, cum_samples)


    def _get_hF_estimate(
            self, pi, Phi, theta, m, eps_expl_a, eps_expl_s, time_limit, 
            max_obs, s_origin, burn_in, 
        ):
        """
        Forms stochastic TD estimator

        :params pi: policy
        :params Phi: feature matrix
        :params m: max mixing time
        :parmas eps_expl_a: action exploration constant
        :parmas eps_expl_s: state exploration constant
        :params time_limit: amount of time left (in sec)
        :params s_origin: see 'estimate_advantage_online_ctd'
        :params burn_in: use burn in to construct hF estimate
        :returns early_terminate: whether time ran out
        :returns hF: stochastic estimator
        :returns cum_samples: total number of samples used
        """
        # setup
        cum_samples = 0
        rand_t = self.rng.geometric(1.-self.gamma)
        s_time = time.time()
        early_terminate = False
        hF = np.zeros(Phi.shape[1])

        if rand_t >= m: 
            return (early_terminate, hF, cum_samples)

        (early_terminate, cum_samples) = self._hF_waiting_period(
            pi, eps_expl_s, s_time, time_limit, max_obs, s_origin, cum_samples
        )
        if early_terminate:
            return (early_terminate, hF, cum_samples)

        # TODO: Combine these two
        if burn_in:
            s_t = self.s
            a_t = self.rng.choice(pi.shape[0], p=pi[:,s_t])
            z_t_idx = s_t*self.n_actions + a_t
            (s_t_next, c_t, _) = self.step(a_t)
            cum_samples += 1

            for t in range(rand_t+1):
                a_t_next = self.rng.choice(pi.shape[0], p=pi[:,s_t_next])

                # form estimator
                z_t_next_idx = s_t_next*self.n_actions + a_t_next
                phi_t = Phi[z_t_idx,:]
                phi_t_next = Phi[z_t_next_idx,:]
                hF_t = phi_t*(phi_t@theta - c_t - self.gamma*phi_t_next@theta)
                hF += (1-self.gamma)*(self.gamma**t) * hF_t

                s_t = s_t_next
                a_t = a_t_next 
                (s_t_next, c_t, _) = self.step(a_t)
                cum_samples += 1

        else:
            for t in range(rand_t):
                a = self.rng.choice(pi.shape[0], p=pi[:,self.s])
                self.step(a)
            cum_samples += rand_t

            # form TD operator
            pi_expl_a = (1.-eps_expl_a)*pi + (eps_expl_a/self.n_actions)
            s_t = self.s
            a_t = self.rng.choice(pi.shape[0], p=pi_expl_a[:,s_t])
            # TODO: Regularization
            (s_t_next, c_t, _) = self.step(a_t)
            a_t_next = self.rng.choice(pi.shape[0], p=pi[:,s_t_next])
            self.step(a_t_next)

            z_t_idx = s_t*self.n_actions + a_t
            z_t_next_idx = s_t_next*self.n_actions + a_t_next
            phi_t = Phi[z_t_idx,:]
            phi_t_next = Phi[z_t_next_idx,:]
            hF = phi_t*(phi_t@theta - c_t - self.gamma*phi_t_next@theta)
            cum_samples += 2 

        return (early_terminate, hF, cum_samples)

    def _most_likely_state(self, pi, N, m, eps_expl_s):
        budget = int(N*m/10) # use 10% of total budget
        state_counter = np.zeros(self.n_states)
        state_counter[self.s] += 1

        pi_expl_s = (1.-eps_expl_s)*pi + (eps_expl_s/self.n_actions)
        for _ in range(budget):
            a = self.rng.choice(pi.shape[0], p=pi_expl_s[:,self.s])
            self.step(a)
            state_counter[self.s] += 1

        return np.argmax(state_counter)

    def _get_hF_minibatch_estimate(self, pi, m, Phi, theta):
        """ takes minibatch of TD operators """
        _, d = Phi.shape
        hF_cum = np.zeros(d)

        for t in range(m):
            # form TD operator
            s_t = self.s
            a_t = self.rng.choice(pi.shape[0], p=pi[:,s_t])
            # TODO: Regularization
            (s_t_next, c_t, term_t) = self.step(a_t)
            a_t_next = self.rng.choice(pi.shape[0], p=pi[:,s_t_next])
            self.step(a_t_next)

            z_t_idx = s_t*self.n_actions + a_t
            z_t_next_idx = s_t_next*self.n_actions + a_t_next
            phi_t = Phi[z_t_idx,:]
            phi_t_next = Phi[z_t_next_idx,:]
            hF_t = phi_t*(phi_t@theta - c_t - self.gamma*phi_t_next@theta)

            hF_cum += (self.gamma**t)*hF_t

        hF_cum *= (1.-self.gamma)

        return (hF_cum, m+1)

    def policy_validation(env, pi, curr_psi, curr_V, settings):
        """
        TODO: Revert so it matches the termination/strong-polynomial paper.

        Evaluates state-action (i.e., advantage) and state value function using
        current policy for `settings["validation_k"]` steps.

        See `policy_validation` below, which may be more sample efficient since it
        uses the random set model.

        :param env: environment from mdpmodel 
        :param pi: policy
        :param curr_psi: current online advantage function
        :param curr_V: current online value function
        :return agg_psi: advantage function 
        :return agg_V: value function 
        :return true_psi: true advantage function
        :return true_V: true value function
        """
        agg_psi = np.zeros((env.n_states, env.n_actions), dtype=float)
        agg_V = np.zeros(env.n_states, dtype=float)
        true_V = -np.inf 
        true_psi = -np.inf*np.ones((env.n_states, env.n_actions), dtype=float)
        if settings["skip_true_model"]:
            (true_psi, true_V) =  env.get_advantage(pi)

        if settings["validation_k"] == 0:
            return agg_psi, agg_V, true_psi, true_V

        if settings["validation_mode"] == "generative":
            for i in range(settings["validation_k"]):
                (psi, V, _) = env.estimate_advantage_generative(pi, settings["N_mc"], settings["T_mc"])
                agg_psi += psi
                agg_V += V
            agg_psi /= N
            agg_V /= N
        elif settings["validation_mode"] == "online_mc_fixed":
            for i in range(settings["validation_k"]):
                (_, psi, V, _) = env.estimate_advantage_online_mc(pi, settings["T_mc"], settings["pi_threshold"])
                agg_psi += psi
                agg_V += V
            agg_psi /= N
            agg_V /= N
        elif settings["validation_mode"] == "random_reset":
            V = env.estimate_random_reset_value(pi, settings["validation_k"])
            psi = env.estimate_random_reset_advantage(pi, settings["validation_k"])
        else:
            warning.warn("Unknown validation mode %s" % settings["validation_mode"])

        return (agg_psi, agg_V, true_psi, true_V)

        # Old code from before (TODO: integrate with above code)
        alpha = float(settings["validation_k"])/(settings["validation_k"] + settings["n_iters"])
        double_agg_V = alpha*agg_V + (1.-alpha)*agg_V_t
        double_agg_psi = alpha*agg_psi + (1.-alpha)*agg_psi_t

        # logger_agg_adv.log(t+1, *list(double_agg_psi.ravel()))
        logger_agg_V.log(t+1, *list(agg_V_t))
        double_agg_advgap = np.maximum(0, np.max(-agg_psi_t, axis=1))
        logger_agg_advgap.log(t+1, *list(double_agg_advgap))

    def policy_validation_random_reset(self, pi, settings):
        """
        Evaluates current policy for `settings["validation_k"]` steps.
        This is different from the `policy_validation` above since we do use a 
        random set model and do not explicitly evaluate all states.

        :param pi: policy
        :param settings: must have key 'max_runtime_in_sec', 'validation_k', and 'skip_true_model'.
        """
        # we only give validation 50% of the max runtime
        validation_time = settings["max_runtime_in_sec"]/2
        V = self.estimate_random_reset_value(pi, settings["validation_k"], validation_time/2)
        psi = self.estimate_random_reset_advantage(pi, settings["validation_k"], validation_time/2)
        true_V = -np.inf * np.ones(self.n_states)
        true_psi = -np.inf*np.ones((self.n_states, self.n_actions), dtype=float)
        V_lb = V - max(0, np.max(-psi))/(1.-self.gamma)
        uni_V_lb = -np.inf

        if settings["skip_true_model"]:
            (true_psi, true_V) = self.get_advantage(pi)
        true_V_rho = np.dot(self.rho, true_V)
        true_V_lb = np.dot(self.rho, true_V - np.max(-true_psi, axis=1)/(1.-self.gamma))
        true_uni_V_lb = np.dot(self.rho, true_V) - np.max(-true_psi)/(1.-self.gamma)
        return (V, V_lb, uni_V_lb, true_V_rho, true_V_lb, true_uni_V_lb)

class KnownModel(MDPModel):
    """ Known (S,A,c,P,gamma) """
    def __init__(self, n_states, n_actions, c, P, gamma, rho=None, seed=None, term_map=None, time_limit=np.inf):
        super().__init__(gamma, seed)

        assert len(c.shape) == 2, "Input cost vector c must be a 2-D vector, recieved %d dimensions" % len(c.shape)
        assert len(P.shape) == 3, "Input cost vector c must be a 3-D tensor, recieved %d dimensions" % len(P.shape)

        assert c.shape[0] == n_states, "1st dimension of c must equal n_states=%d, was instead %d" % (n_states, c.shape[0])
        assert c.shape[1] == n_actions, "2nd dimension of c must equal n_actions=%d, was instead %d" % (n_actions, c.shape[1])
        assert P.shape[0] == n_states, "1st dimension of P must equal n_states=%d, was instead %d" % (n_states, P.shape[0])
        assert P.shape[1] == n_states, "2nd dimension of P must equal n_states=%d, was instead %d" % (n_states, P.shape[1])
        assert P.shape[2] == n_actions, "3rd dimension of P must equal n_actions=%d, was instead %d" % (n_actions, P.shape[2])

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
        self.term_map = term_map
        self.gamma = gamma
        self.time_limit=time_limit
        if rho is None:
            rho = np.ones(self.n_states, dtype=float)/self.n_states
        self.rho = rho

        # initialize a 
        self.s = self.rng.integers(0, self.n_states)
        self.t = 0

        # initialize rbf for solving with linear function approx
        self.init_linear = False

        # initialize for tracking episode cost
        self.curr_ep_len = 0
        self.ep_cum_cost = 0.
        self.ep_cum_cost_arr = np.zeros(1024, dtype=float)
        self.ep_len_arr = np.zeros(1024, dtype=int)
        self.ep_cum_samps_arr = np.zeros(1024, dtype=int)
        self.ep_ct = 0

    def step(self, a):
        c = self.c[self.s, a]
        curr_s = self.s
        self.s = self.rng.choice(self.P.shape[0], p=self.P[:,self.s, a])
        terminated = 0 if (self.term_map is None) else self.term_map[self.s, curr_s, a]
        self.ep_cum_cost += self.gamma**(self.curr_ep_len) * c
        self.t += 1
        self.curr_ep_len += 1

        if self.curr_ep_len % self.time_limit == 0:
            terminated = True

        if terminated:
            self.ep_cum_cost_arr[self.ep_ct] = self.ep_cum_cost
            self.ep_len_arr[self.ep_ct] = self.curr_ep_len
            self.ep_cum_samps_arr[self.ep_ct] = \
                self.ep_cum_samps_arr[max(0, self.ep_ct-1)] + self.curr_ep_len

            self.ep_ct += 1
            if self.ep_ct == len(self.ep_cum_cost_arr):
                self.ep_cum_cost_arr = np.append(
                    self.ep_cum_cost_arr, 
                    np.zeros(self.ep_ct, dtype=self.ep_cum_cost_arr.dtype)
                )
                self.ep_len_arr = np.append(
                    self.ep_len_arr, 
                    np.zeros(self.ep_ct, dtype=self.ep_len_arr.dtype)
                )
                self.ep_cum_samps_arr = np.append(
                    self.ep_cum_samps_arr, 
                    np.zeros(self.ep_ct, dtype=self.ep_cum_samps_arr.dtype)
                )

            self.ep_cum_cost = 0
            self.curr_ep_len = 0

        return (self.s, c, terminated)

    def get_cum_cost_and_len_arr(self):
        return (
            self.ep_cum_cost_arr[:self.ep_ct], 
            self.ep_len_arr[:self.ep_ct],
            self.ep_cum_samps_arr[:self.ep_ct],
        )

    def get_stationary(self, pi):
        P_pi = np.einsum('psa,as->ps', self.P, pi)
        P_prime = np.vstack((np.eye(self.n_states) - P_pi, np.ones(self.n_states)))
        nu = la.lstsq(P_prime, np.append(np.zeros(self.n_states), 1))[0]
        # normalize
        nu = np.maximum(nu, 0)
        nu = nu/np.sum(nu)

        # check reversibility
        # la.norm(np.diag(nu)*(P_pi - P_pi.T))

        return nu

    def get_discounted_visitation(self, pi, mu):
        P_pi = np.einsum('psa,as->ps', self.P, pi)
        kappa = (1-self.gamma)*la.solve(np.eye(self.n_states) - self.gamma*P_pi, mu)
        # normalize
        kappa -= np.min(kappa)
        kappa /= np.sum(kappa)
        return kappa

    def _get_spectral_gap(self, pi):
        """
        Spectral gap is 1-max(|lam_2|,|lam_n|)
        """
        P_pi = np.einsum('psa,as->ps', self.P, pi) 
        eigvals = np.sort(la.eig(P_pi)[0].real)[::-1]
        if len(eigvals) == 1:
            return 1e-6
        return 1. - max(abs(eigvals[1]), abs(eigvals[-1]))

    def get_mixing_time_ub(self, pi):
        """ Use

            $$(t_relax-1)*ln(2) <= t_mix <= t_relax*ln(4/nu_*),$$

        where nu_* is the smallest steady state distribution value and t_relax=1/(spec_gap).
        """
        spec_gap = self._get_spectral_gap(pi)
        nu = self.get_stationary(pi)

        if np.min(nu) == 0:
            return np.inf, nu

        return np.log(4/np.min(nu))/spec_gap, nu

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

    def estimate_advantage_generative(self, pi, N, T):
        """
        Uses matrix-vector tricks to speedup the generative model

        :param N: number of Monte Carlo simulations to run per state-action pair
        :param T: duration to for each Monte Carlo simulation
        """
        # 1 x S
        pi_sum = np.cumsum(pi, axis=0)
        # S x (SA)
        P_reshape = np.reshape(self.P, newshape=(self.P.shape[0], self.P.shape[1]*self.P.shape[2]))
        P_reshape_sum = np.cumsum(P_reshape, axis=0)

        # SA
        q = np.zeros(self.n_states*self.n_actions, dtype=float)

        for i in range(N):
            s_arr = np.kron(np.arange(self.n_states), np.ones(self.n_actions, dtype=int))
            a_arr = np.kron(np.ones(self.n_states, dtype=int), np.arange(self.n_actions))
            for t in range(T):
                q += self.gamma**t * self.c[s_arr, a_arr]

                u = self.rng.uniform(size=len(q))
                z_arr = s_arr * self.n_actions + a_arr
                s_arr = np.argmax(np.outer(u, np.ones(self.n_states)) < P_reshape_sum[:,z_arr].T, axis=1)

                u = self.rng.uniform(size=len(q))
                a_arr = np.argmax(np.outer(u, np.ones(self.n_actions)) < pi_sum[:,s_arr].T, axis=1)

        q /= N
        Q = np.reshape(q, newshape=(self.n_states, self.n_actions))

        V_pi = np.einsum('sa,as->s', Q, pi)
        psi = Q - np.outer(V_pi, np.ones(self.n_actions, dtype=float))
        total_samples = N*self.n_states*self.n_actions*T

        return (psi, V_pi, total_samples)
                    
    def get_steadystate(self, pi):
        P_pi = np.einsum('psa,as->ps', self.P, pi)

        dim = P_pi.shape[0]
        Q = (P_pi.T-np.eye(dim))
        ones = np.ones(dim)
        Q = np.c_[Q,ones]
        QTQ = np.dot(Q, Q.T)

        # check singular
        try:
            if la.matrix_rank(QTQ) < QTQ.shape[0]:
                print("Singular matrix when computing stationary distribution, return zero vector")
                return np.zeros(QTQ.shape[0], dtype=float)
        except:
            # error with matrix rank
            return np.zeros(QTQ.shape[0], dtype=float)

        bQT = np.ones(dim)
        return np.linalg.solve(QTQ,bQT)

class Bandits(KnownModel):
    def __init__(self, n_arms, gamma, seed):
        n_actions = n_arms
        n_states = 1

        P = np.ones((n_states, n_states, n_actions), dtype=float)
        c = np.zeros((n_states, n_actions), dtype=float)
        c[0,:] = np.linspace(0,1,num=n_arms,endpoint=True)
        rho = np.ones(1)

        super().__init__(n_states, n_actions, c, P, gamma, rho, seed)

class GridWorldWithTraps(KnownModel):
    def __init__(self, length, n_traps, gamma, n_origins=-1, eps=0.05, time_limit=1024, seed=None, ergodic=False, low_dim=-1):
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
        n_targets = 1
        if n_origins == -1:
            n_origins = n_states-n_traps-n_targets

        # have the same set of traps, origins, and traps
        valid_points = np.arange(length*length)
        if low_dim > 0:
            valid_points_original = np.arange(low_dim * low_dim)
            valid_points_proj_y = (valid_points_original // low_dim).astype('int')
            valid_points_proj_x = (valid_points_original % low_dim).astype('int')
            valid_points = valid_points_proj_y * length + valid_points_proj_x

        rng = np.random.default_rng(0)
        n_requested_pts = n_traps+n_targets+n_origins
        n_rnd_pts = min(len(valid_points), n_requested_pts)
        rnd_pts = rng.choice(valid_points, replace=False, size=n_rnd_pts)

        traps = rnd_pts[:n_traps]
        self.origins = rnd_pts[n_traps:n_traps+n_origins]
        rho = np.zeros(length*length, dtype=float)
        rho[self.origins] = 1./len(self.origins)
        self.target = target = rnd_pts[-1]
        print("==== ENV INFO ====")
        print("  Target at index %d" % target)
        print("  Traps at ", np.sort(traps))
        if len(self.origins) < 10:
            print("  Origins at ", np.sort(self.origins))

        P = np.zeros((n_states, n_states, n_actions), dtype=float)
        c = np.zeros((n_states, n_actions), dtype=float)
        term_map = np.zeros((n_states, n_states, n_actions), dtype=int)

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
            # rnd_pts = rng.choice(length*length, replace=False, size=n_traps+1)
            # non_target_nor_trap = np.setdiff1d(np.arange(length*length), rnd_pts)

            P[:,target,:] = 0
            # go to random non-target non-trap location
            P[self.origins,target,:] = 1./len(self.origins)
            term_map[self.origins,target,:] = 1
        else:
            P[:,target,:] = 0
            # stay at target
            P[target,target,:] = 1.
            term_map[target,target,:] = 1

        # apply trap cost
        c[:,:] = 1.
        c[traps,:] = 10.
        c[target,:] = -10.

        super().__init__(n_states, n_actions, c, P, gamma, rho, seed, term_map=term_map, time_limit=time_limit)

    def get_target(self):
        return self.target

    def step(self, a):
        (s, c, terminated) = super().step(a)

        # means we just reset due to termination (either time limit or reach target)
        if terminated:
            self.s = self.rng.choice(self.origins)

        return (s, c, terminated)

class GridWorldWithTrapsAndHills(KnownModel):
    def __init__(self, length, n_traps, gamma, eps=0.05, seed=None, ergodic=False):
        """ Same 2D gridworld, but the probability of moving towards the target
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

        super().__init__(n_states, n_actions, c, P, gamma, seed=seed)

    def get_target(self):
        return self.target

class Taxi(KnownModel):
    # R, Y, G, B (x,y)
    color_arr = [(0,0), (0,4), (4,0), (3,4)]

    right_wall_arr = [(1,0), (1,1), (0,3), (0,4), (2,3), (2,4)]
    left_wall_arr  = [(2,0), (2,1), (1,3), (1,4), (3,3), (3,4)]

    def __init__(self, gamma, eps=0., n_origins=-1, ergodic=False, seed=None):
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
        if n_origins == -1:
            n_origins = 5*5*4*3 # all possible places except when passenger is in taxi or destination

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

        rng = np.random.default_rng(0)
        starting_states = rng.choice(starting_states, size=min(n_origins, len(starting_states)), replace=False)

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

            c[curr_state_loc, 5] = -20

        super().__init__(n_states, n_actions, c, P, gamma, seed=seed)

class Random(KnownModel):
    def __init__(self, n_states, n_actions, gamma, seed=None):
        rng = np.random.default_rng(seed)
        P = rng.integers(0, int(1e6), size=(n_states, n_states, n_actions)).astype(float)
        sum_P = np.sum(P, axis=0)
        P /= sum_P
        c = rng.normal(size=(n_states, n_actions))

        super().__init__(n_states, n_actions, c, P, gamma)

class Small(KnownModel):
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

class Chain(KnownModel):
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

class SimpleBattery(KnownModel):
    def __init__(self, n_solar_pwr, n_battery_change_lim, n_price_pts, gamma, rho=None, seed=None):
        """ Simple finite state finite action battery model with three elements:

        - Solar panel - with n_solar_pwr different power points
        - Energy storage - with [0, 2*n_solar_pwr + 4*battery_change_lim] storage settings)
        - Grid - with n_price_pts different price points (around 10% will be negative)

        The solar panel and grid change according to their own Markov chain.
        The energy storage follows basic flow equations under the following constraints:

        1. at most [-battery_change_lim,battery_change_lim] change in energy for the battery
        2. if battery change is zero, all solar energy is sold
        3. if battery change is positive, we first use solar energy before buying from grid
        4. if battery change is negative, we also sell all solar energy.

        The cost (which we want to minimize over time) at each step is

            c_t = energy_price * battery_change
        """

        n_storage_pts = 2*n_solar_pwr + 4*n_battery_change_lim
        storage_arr = np.arange(n_storage_pts)
        max_storage = np.max(storage_arr)
        solar_pwr_arr = np.arange(n_solar_pwr)
        n_neg_prices = max(1, int(0.1*n_price_pts))
        grid_price_arr = np.arange(-n_neg_prices, -n_neg_prices + n_price_pts)
        battery_change_arr = np.arange(-n_battery_change_lim,n_battery_change_lim+1)

        n_states = n_solar_pwr * n_price_pts * n_storage_pts
        n_actions = len(battery_change_arr)

        print("==== ENV INFO ====")
        print("  storage values: ", storage_arr)
        print("  solar values: ", solar_pwr_arr)
        print("  price values: ", grid_price_arr)

        P = np.zeros((n_states, n_states, n_actions), dtype=float)
        c = np.zeros((n_states, n_actions), dtype=float)

        def get_substate_index(i):
            i_solar = i // (n_price_pts * n_storage_pts)
            i_price = i // (n_storage_pts)
            i_storage = i % n_storage_pts
            return (i_solar, i_price, i_storage)

        for x,a in enumerate(battery_change_arr):
            for i in range(n_states):
                (i_solar, i_price, i_storage) = get_substate_index(i)
                # only enumerate all possible values of solar and price, 
                # since next storage is fixed once we know current state and action
                for j in range(n_solar_pwr * n_price_pts):
                    (j_solar, j_price, _) = get_substate_index(j * n_storage_pts)
                    prob_solar_change = 1./(abs(i_solar-j_solar)+1)
                    prob_price_change = 1./(abs(i_price-j_price)+1)

                    next_storage = np.clip(i_storage + a, 0, max_storage)
                    i_next = j * n_storage_pts + next_storage
                    # joint distribution of solar and price change
                    P[i_next,i,a] = prob_solar_change * prob_price_change 

                    # negative is we sold energy
                    energy_change = (next_storage - i_storage) - i_solar
                    c[i,a] = energy_change * i_price

        # normalize
        for i in range(P.shape[1]):
            for j in range(P.shape[2]):
                P[:,i,j] = P[:,i,j]/np.sum(P[:,i,j])
        c -= np.min(c)
        # c /= np.max(c)

        if rho is None:
            rho = np.ones(n_states, dtype=float)/n_states
        self.rho = rho
            
        super().__init__(n_states, n_actions, c, P, gamma, rho, seed)

    def get_target(self):
        return self.target

class DiscretizedGymnasiumModel(MDPModel):
    """
    We form an approximate Gymnasium model by discretizing the state space.
    """
    def __init__(self, env_name, gamma, resolution, seed=None, max_space_val=np.inf):
        super().__init__(gamma, seed)
        # self.env = gym.make(env_name, render_mode="human")
        self.env = gym.make(env_name)

        assert isinstance(self.env.action_space, gym.spaces.discrete.Discrete), "Discretized env's act. space must be discrete (was %s)" % type(self.env.action_space)
        assert isinstance(self.env.observation_space, gym.spaces.box.Box), "Space %s cannot be discretized" % type(self.env.observation_space)

        self.low, self.high = self.env.observation_space.low, self.env.observation_space.high
        self.low = np.maximum(-max_space_val, self.low)
        self.high = np.minimum(max_space_val, self.high)
        self.diff = self.high - self.low
        state_dim = len(self.low)
        self.state_flatten_mult_arr = np.power(resolution, np.arange(state_dim)[::-1])
        self.resolution = resolution
        self.n_states = resolution**state_dim
        self.n_actions = self.env.action_space.n
        self.ct = seed

        s_cont, _ = self.env.reset(seed=self.ct) 
        self.s = self.state_discretize(s_cont)
        self.s_0 = np.copy(self.s)
        self.ct += 1

        self.rho = np.zeros(self.n_states)
        self.rho[self.s] = 1

    def get_origin_state(self):
        return (self.state_discretize(self.s_0), self.s_0)

    def state_discretize(self, s_cont):
        s_bucket = np.round(np.divide(s_cont - self.low, self.diff) * self.resolution)
        return int(np.dot(s_bucket, self.state_flatten_mult_arr))

    def step(self, a):
        s_cont, r, terminated, truncated, _ = self.env.step(a)
        terminated = terminated or truncated

        # see: https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
        if terminated:
            s_cont, _ = self.env.reset(seed=self.ct) 
            self.ct += 1

        self.s = self.state_discretize(s_cont)

        return (self.s, -r, terminated)

    def get_mixing_time_ub(self, pi):
        return 0, np.zeros(1)

class Garnet(KnownModel):
    """
    src: https://proceedings.mlr.press/v89/tarbouriech19a.html

    5 parameters: (S,A,b,sig_min_sq,sig_max_sq)
       - (S,A): state action space 
       - b: branching factor, is number of next states transitions, where the probability is from Uni[0,1]
       - [sig_min_sq, sig_max_sq]: variance of cost at each c(s,a) - assume same mean of mu = 0.5
    """
    def __init__(self, n_states, n_actions, gamma, b, sig_min_sq, sig_max_sq, time_limit_mult=20, seed=None):
        rng = np.random.default_rng(seed)
        n_states_remove = int((1.-b) * n_states)
        mu = 0

        P = rng.uniform(size=((n_states, n_states, n_actions)))
        for i in range(n_states*n_actions):
            (s,a) = (i//n_actions, i%n_actions)
            idxs = rng.choice(n_states, size=n_states_remove, replace=False)
            P[idxs,s,a] = 0
            P[s,s,a] = 0.001
            P[:,s,a] /= np.sum(P[:,s,a])

        sigs_arr = np.sqrt(rng.uniform(sig_min_sq, sig_max_sq, size=((n_states, n_actions))))
        c = rng.normal(0, scale=sigs_arr)
        c = (c-np.min(c))/(np.max(c)-np.min(c))
        time_limit = int(time_limit_mult/(1.-gamma))

        super().__init__(n_states, n_actions, c, P, gamma, time_limit=time_limit)

class BlackJack(KnownModel):
    """
    Based on: https://gymnasium.farama.org/environments/toy_text/blackjack/

    Unlike the Gymnasium implementation, we do not use the usable-ace state since
    this lead to some state never visited (e.g., value of 1 from ACE but no usable ace)
    """
    # TODO: Logic is tricky due to 1) face cards 2) usable ACE 3) draw until 17
    def __init__(self):
        # value can be as low as 4 and as high as as 31 
        # from blackjack + face card (drawing an ace counts as 1 since it leads a bust)
        n_states = 30*11
        n_actions = 2
        init_cards = np.arange(4, 22)
        all_cards = np.arange(2,12) * 11
        all_cards_usable = np.arange(1,11) * 11

        init_dist = np.append(np.arange(1, 11), np.arange(2,10)[::-1])
        init_dist[8] += 1 # for double aces (with value 12)
        init_dist /= np.sum(init_dist)
        # 3 face cards
        all_dist = np.ones(len(all_cards)); all_dist[-2] = 3; all_dist /= np.sum(all_dist)
        all_usable_dist = np.ones(len(all_cards)); all_usable_dist[-1] = 3; all_usable_dist /= np.sum(all_usable_dist)

        P = rng.zeros(size=((n_states, n_states, n_actions)))

        # if hit (a = 1)
        for i in range(n_states):
            val = i // 11
            bust = val > 21
            cost = 0

            if bust: # reset
                next_vals = init_cards
                next_dist = init_dist
                cost = 1 # large cost
            elif val > 10: # drawing ace would bust
                next_vals = val + all_cards_usable
                next_dist = all_usable_dist
            else: # drawing ace would not bust
                next_vals = val + all_cards
                next_dist = all_dist
            P[next_vals, i, 1] = next_dist
            c[i, 1] = cost

        # if stick (a = 0)
        P[all_cards, :, 0] = 0.1

        sigs_arr = np.sqrt(rng.uniform(sig_min_sq, sig_max_sq, size=((n_states, n_actions))))
        c = rng.normal(0, scale=sigs_arr)
        c = (c-np.min(c))/(np.max(c)-np.min(c))

        super().__init__(n_states, n_actions, c, P, gamma)

def get_env(name, gamma, seed=None):
    if name == "bandits":
        env = Bandits(4, gamma, seed=seed)
    elif name == "gridworld_footnote":
        env = GridWorldWithTraps(5, 3, gamma, seed=seed, ergodic=True)
    # TEMP
    elif name == "gridworld_verytiny":
        env = GridWorldWithTraps(2, 0, gamma, seed=seed, ergodic=True)
    elif name == "gridworld_tiny":
        env = GridWorldWithTraps(10, 5, gamma, seed=seed, ergodic=True)
    elif name == "gridworld_small":
        env = GridWorldWithTraps(20, 20, gamma, seed=seed, ergodic=True)
    elif name == "gridworld_large":
        env = GridWorldWithTraps(50, 50, gamma, seed=seed, ergodic=True)
    elif name == "gridworld_small_low_dim":
        env = GridWorldWithTraps(20, 20, gamma, seed=seed, ergodic=True, low_dim=10, time_limit=128)
    elif name == "gridworld_large_low_dim":
        env = GridWorldWithTraps(50, 50, gamma, seed=seed, ergodic=True, low_dim=10, time_limit=128)
    elif name == "gridworld_hill_small":
        env = GridWorldWithTrapsAndHills(20, 20, gamma, seed=seed, ergodic=True)
    elif name == "gridworld_hill_large":
        env = GridWorldWithTrapsAndHills(50, 50, gamma, seed=seed, ergodic=True)
    elif name == "gridworld_footnote_loop":
        env = GridWorldWithTraps(5, 3, gamma, seed=seed, ergodic=True, n_origins=1)
    elif name == "gridworld_tiny_loop":
        env = GridWorldWithTraps(10, 5, gamma, seed=seed, ergodic=True, n_origins=1)
    elif name == "gridworld_small_loop":
        env = GridWorldWithTraps(20, 20, gamma, seed=seed, ergodic=True, n_origins=1)
    elif name == "gridworld_large_loop":
        env = GridWorldWithTraps(50, 50, gamma, seed=seed, ergodic=True, n_origins=1)
    elif name == "taxi":
        env = Taxi(gamma, ergodic=True)
    elif name == "taxi_sparse":
        env = Taxi(gamma, ergodic=True, n_origins=5)
    elif name == "random":
        env = Random(100, 100, gamma, seed=seed)
    elif name == "chain":
        env = Chain(100, gamma, eps=1e-3, seed=seed)
    elif name == "battery":
        env = SimpleBattery(3, 2, 4, gamma, seed=seed)
    elif name == "discrete_mountaincar":
        env = DiscretizedGymnasiumModel("MountainCar-v0", gamma, 100, seed=seed)
    elif name == "discrete_cartpole":
        env = DiscretizedGymnasiumModel("CartPole-v1", gamma, 100, seed=seed, max_space_val=100.)
    elif name == "garnet_50":
        env = Garnet(50, 5, gamma, 0.2, 0.5, 2.0, seed=seed)
    elif name == "garnet_100":
        env = Garnet(100, 10, gamma, 0.2, 0.5, 2.0, seed=seed)
    elif name == "garnet_200":
        env = Garnet(200, 30, gamma, 0.2, 0.5, 2.0, seed=seed)
    elif name == "garnet_500":
        env = Garnet(500, 30, gamma, 0.2, 0.5, 2.0, seed=seed)
    elif name == "garnet_1000":
        env = Garnet(1000, 30, gamma, 0.2, 0.5, 2.0, seed=seed)
    elif name == "garnet_2500":
        env = Garnet(2500, 30, gamma, 0.2, 0.5, 2.0, seed=seed)
    else:
        raise Exception("Unknown env_name=%s" % name)

    return env

