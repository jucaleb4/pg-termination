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

        has_adjusted_time = False
        T_time_adjusted = np.inf
        # run Monte Carlo while estimating 1 percentage of total runtime
        for t in range(T):
            states[t] = self.s
            actions[t] = self.rng.choice(pi.shape[0], p=pi[:,states[t]])
            (_, costs[t]) = self.step(actions[t])

            # early termination due to time
            if (not has_adjusted_time) and ((time.time() - s_time) >= 0.01 * time_limit):
                has_adjusted_time = True
                # increase 67X due acct for Q (assume sampling is ~2/3 of time)
                T_time_adjusted = int(min(T, 67*(t+1)))
            elif t > T_time_adjusted:
                T = t; break
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

        return (psi, V_pi, visit_len_state_action, T)

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
                T = t+1
                break

        # normalize and compute intermediates
        if np.min(nu_est) == 0:
            nu_est += 1e-6 # small perturbation
        nu_est /= T; M_est /= (T-1)
        nu_est_invsq = np.reciprocal(np.sqrt(nu_est))
        L = np.diag(nu_est_invsq)@M_est@np.diag(nu_est_invsq)
        sym_L = 0.5*(L + L.T)
        eig_sym_L = np.sort(la.eig(sym_L)[0])[::-1]

        # estimate spectral gap
        # if we have incorrect value, use heuristics
        nu_est_lb = np.min(nu_est) 
        spec_gap_est = 1 - max(eig_sym_L[1], abs(eig_sym_L[-1]))
        if (not (0 < spec_gap_est < 1)):
            eig_L = np.sort(la.eig(L)[0])[::-1]
            spec_gap_est = 1 - max(eig_L[1], abs(eig_L[-1]))
        elif (not (0 < spec_gap_est < 1)):
            spec_gap_est = 1e-3
        trelax_est = 1./spec_gap_est
        tmix_est = trelax_est * np.log(4/nu_est_lb)

        n_samples = T
        return (nu_est, tmix_est, n_samples)

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

        t = 0
        s_time = time.time()
        has_adjusted_time = False
        T_time_adjusted = np.inf

        countdown = max(1, np.log(1./eps))
        start_countdown = False
        while countdown > 0:
            states[t] = self.s
            actions[t] = self.rng.choice(pi.shape[0], p=pi[:,states[t]])
            (_, costs[t]) = self.step(actions[t])
            unvisited_sa[actions[t], states[t]] = 0

            # check if we can terminate
            if t % cycle == 0:
                start_countdown = (np.max(unvisited_sa) == 0)
            if start_countdown:
                countdown -= 1

            # early termination due to time
            if (not has_adjusted_time) and ((time.time() - s_time) >= 0.01 * time_limit):
                has_adjusted_time = True
                T_time_adjusted = int(67*(t+1))
            elif t > T_time_adjusted:
                t += 1
                break
            elif (t+1) == len(costs):
                costs = np.append(costs, np.zeros(len(costs)))
                states = np.append(states, np.zeros(len(states), dtype=int))
                actions = np.append(actions, np.zeros(len(actions), dtype=int))
            t += 1

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

        # for proabibilities that are very low, set Q value to be high
        (poor_sa_a, poor_sa_s) = np.where(pi <= threshold)
        Q_max = np.max(np.abs(self.c))/(1.-self.gamma)
        Q[poor_sa_s,poor_sa_a] = Q_max

        V_pi = np.einsum('sa,as->s', Q, pi)
        psi = Q - np.outer(V_pi, np.ones(self.n_actions, dtype=float))

        return (psi, V_pi, visit_len_state_action, T)

    def estimate_advantage_online_ctd(self, pi, Phi, ukappa, eps, delta, 
            iota_mult, is_finite_state, prev_theta=None
    ):
        """
        Forms nearly unbiased TD estimator that is bounded w.h.p.

        :params pi: policy
        :params Phi: feature matrix
        :params ukappa: lower bound estimate of minimum discounted visit distribution
        :params eps: accuracy tolerance
        :params delta: failure rate
        :params iota_mult: multiplier for iota (stepsize)
        :params is_finite_state: whether state space is finite
        """

        # initialize data and parameters
        cum_samples = 0
        uw = (1.-self.gamma)*ukappa**2/self.n_actions 
        # if is_finite_state:
        #     Phi_sigvals = la.svd(Phi)[1] 
        # else:
        #     # TODO: Find more elegant way to do this in continuous space
        #     Phi_sigvals = la.svd(Phi[0])[1] 
        Omega = la.norm(Phi, ord=2)
        # skip smallest mu
        mu = min(1, Omega)**2*uw # np.min(Phi_sigvals)**2*uw
        T  = Omega**2/((1.-self.gamma)**2*mu)
        iota = (1.-self.gamma)/Omega**2 
        N = int((T/uw) * max(1, np.log(1./eps))) 
        # m = int(np.log(1./(uw*eps)**2)/np.log(1/self.gamma)) 
        # m = int(10/np.log(2)) 
        m = int(1./(1.-self.gamma))
        eps_expl_s = 1
        eps_expl_a = (1.-self.gamma)*min(1,ukappa)/4
        # robust_trials = min(5, np.log(1./delta)/np.log(2))
        robust_trials = 1

        iota *= iota_mult

        hQ_arr = np.zeros((Phi.shape[0], robust_trials))
        for j in range(robust_trials):
            (Q_est, n_samples, theta) = self._nonrobust_ctd(
                pi, Phi, iota, N, m, eps_expl_a, eps_expl_s, prev_theta)
            hQ_arr[:,j] = Q_est
            cum_samples += n_samples

        hQ_arr_norms = la.norm(hQ_arr, ord=np.inf, axis=0)
        j_star = np.argmin(hQ_arr_norms)

        robust_hQ = hQ_arr[:,j_star]
        robust_Q = np.reshape(robust_hQ, newshape=(self.n_states, self.n_actions), order='C')
        robust_V = np.einsum('sa,as->s', robust_Q, pi)
        robust_psi = robust_Q - np.outer(robust_V, np.ones(self.n_actions))

        return (robust_psi, robust_V, cum_samples, theta)

    def _nonrobust_ctd(self, pi, Phi, iota, N, m, eps_expl_a, eps_expl_s, prev_theta):
        """
        Forms nearly unbiased TD estimator with bounded second moment.
        """
        _, d = Phi.shape
        theta_t = np.zeros(d) if prev_theta is None else prev_theta 
        cum_samples = 0

        # estimate most likely state
        s_origin = self._most_likely_state(pi, N, m, eps_expl_s)

        for t in range(N):
            (hF_t, n_samples) = self._get_hF_estimate(pi, m, s_origin, eps_expl_a, 
                                                    eps_expl_s, Phi, theta_t)
            # (hF_t, n_samples) = self._get_hF_minibatch_estimate(pi, m, Phi, theta_t)
            theta_t = theta_t - iota*hF_t
            cum_samples += n_samples

        latter_avg_theta = np.zeros(d)
        for t in range(N):
            (hF_t, n_samples) = self._get_hF_estimate(pi, m, s_origin, eps_expl_a, 
                                                    eps_expl_s, Phi, theta_t)
            # (hF_t, n_samples) = self._get_hF_minibatch_estimate(pi, m, Phi, theta_t)
            theta_t = theta_t - iota*hF_t
            alpha_t = 1./(t+1)
            latter_avg_theta = (1.-alpha_t)*latter_avg_theta + alpha_t*theta_t
            cum_samples += n_samples
            
        hQ = Phi@latter_avg_theta
        return (hQ, cum_samples, latter_avg_theta)

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

    def _get_hF_estimate(self, pi, m, s_origin, eps_expl_a, eps_expl_s, Phi, theta):
        # rand_t = self.rng.geometric(1.-self.gamma)
        rand_t = self.rng.geometric(0.5)
        _, d = Phi.shape
        hF = np.zeros(d)
        # if rand_t >= m:
        #     print('skipped!')
        #     return (np.zeros(d), 0)

        # pi_expl_s = (1.-eps_expl_s)*pi + (eps_expl_s/self.n_actions)
        # while self.s != s_origin:
        #     a = self.rng.choice(pi.shape[0], p=pi_expl_s[:,self.s])
        #     self.s = self.rng.choice(self.P.shape[0], p=self.P[:,self.s,a])

        for t in range(rand_t):
            a = self.rng.choice(pi.shape[0], p=pi[:,self.s])
            self.step(a)

        # form TD operator
        pi_expl_a = (1.-eps_expl_a)*pi + (eps_expl_a/self.n_actions)
        s_t = self.s
        a_t = self.rng.choice(pi.shape[0], p=pi_expl_a[:,s_t])
        # TODO: Regularization
        (s_t_next, c_t) = self.step(a_t)
        a_t_next = self.rng.choice(pi.shape[0], p=pi[:,s_t_next])
        self.step(a_t_next)

        rand_t += 2 
        z_t_idx = s_t*self.n_actions + a_t
        z_t_next_idx = s_t_next*self.n_actions + a_t_next
        phi_t = Phi[z_t_idx,:]
        phi_t_next = Phi[z_t_next_idx,:]
        hF = phi_t*(phi_t@theta - c_t - self.gamma*phi_t_next@theta)

        return (hF, rand_t)

    def _get_hF_minibatch_estimate(self, pi, m, Phi, theta):
        """ takes minibatch of TD operators """
        _, d = Phi.shape
        hF_cum = np.zeros(d)

        for t in range(m):
            # form TD operator
            s_t = self.s
            a_t = self.rng.choice(pi.shape[0], p=pi[:,s_t])
            # TODO: Regularization
            (s_t_next, c_t) = self.step(a_t)
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

class KnownModel(MDPModel):
    """ Known (S,A,c,P,gamma) """
    def __init__(self, n_states, n_actions, c, P, gamma, rho=None, seed=None):
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
        self.gamma = gamma
        if rho is None:
            rho = np.ones(self.n_states, dtype=float)/self.n_states
        self.rho = rho

        # initialize a 
        self.s = self.rng.integers(0, self.n_states)

        # initialize rbf for solving with linear function approx
        self.init_linear = False

    def step(self, a):
        c = self.c[self.s, a]
        self.s = self.rng.choice(self.P.shape[0], p=self.P[:,self.s, a])
        return (self.s, c)

    def get_stationary(self, pi):
        P_pi = np.einsum('psa,as->ps', self.P, pi)
        P_prime = np.vstack((np.eye(self.n_states) - P_pi, np.ones(self.n_states)))
        nu = la.lstsq(P_prime, np.append(np.zeros(self.n_states), 1))[0]
        # normalizeg
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

    def estimate_advantage_generative_slow(self, pi, N, T):
        """
        :param N: number of Monte Carlo simulations to run per state-action pair
        :param T: duration to for each Monte Carlo simulation
        """
        Q = np.zeros((self.n_states, self.n_actions), dtype=float)

        for s in range(self.n_states):
            for a in range(self.n_actions):
                costs = 0.
                for i in range(N):
                    s_t = s
                    a_t = a
                    for t in range(T):
                        Q[s,a] += self.gamma**t * self.c[s_t,a_t]
                        s_t_next = self.rng.choice(self.P.shape[0], p=self.P[:,s_t,a_t])
                        a_t = self.rng.choice(pi.shape[0], p=pi[:,s_t])
                        s_t = s_t_next

                Q[s,a] /= N

        V_pi = np.einsum('sa,as->s', Q, pi)
        psi = Q - np.outer(V_pi, np.ones(self.n_actions, dtype=float))

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
    def __init__(self, length, n_traps, gamma, n_origins=-1, eps=0.05, seed=None, ergodic=False):
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
        if n_origins == -1:
            n_origins = n_states-n_traps-1

        # have the same set of traps, origins, and traps
        rng = np.random.default_rng(0)
        rnd_pts = rng.choice(length*length, replace=False, size=n_traps+1+n_origins)

        rng = np.random.default_rng(seed)
        traps = rnd_pts[:n_traps]
        origins = rnd_pts[n_traps:n_traps+n_origins]
        rho = np.zeros(length*length, dtype=float)
        rho[origins] = 1./len(origins)
        self.target = target = rnd_pts[-1]
        print("==== ENV INFO ====")
        print("  Target at index %d" % target)
        print("  Traps at ", np.sort(traps))
        if len(origins) < 10:
            print("  Origins at ", np.sort(origins))

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
            # rnd_pts = rng.choice(length*length, replace=False, size=n_traps+1)
            # non_target_nor_trap = np.setdiff1d(np.arange(length*length), rnd_pts)

            P[:,target,:] = 0
            # go to random non-target non-trap location
            P[origins,target,:] = 1./len(origins)
        else:
            P[:,target,:] = 0
            # stay at target
            P[target,target,:] = 1.

        # apply trap cost
        c[:,:] = 1.
        c[traps,:] = 10.
        c[target,:] = -10.

        super().__init__(n_states, n_actions, c, P, gamma, rho, seed)

    def get_target(self):
        return self.target

class GridWorldWithTrapsAndHills(KnownModel):
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
    def __init__(self, env_name, gamma, resolution, seed=None):
        super().__init__(gamma, seed)
        # self.env = gym.make(env_name, render_mode="human")
        self.env = gym.make(env_name)

        self.low, self.high = self.env.observation_space.low, self.env.observation_space.high
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

        # see: https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
        if terminated or truncated:
            s_cont, _ = self.env.reset(seed=self.ct) 
            self.ct += 1

        self.s = self.state_discretize(s_cont)

        return (self.s, -r)

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
    def __init__(self, n_states, n_actions, gamma, b, sig_min_sq, sig_max_sq, seed=None):
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

        super().__init__(n_states, n_actions, c, P, gamma)

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
    elif name == "gridworld_tiny":
        env = GridWorldWithTraps(10, 5, gamma, seed=seed, ergodic=True)
    elif name == "gridworld_small":
        env = GridWorldWithTraps(20, 20, gamma, seed=seed, ergodic=True)
    elif name == "gridworld_small_sparse":
        env = GridWorldWithTraps(20, 20, gamma, seed=seed, ergodic=True, n_origins=1)
    elif name == "gridworld_large":
        env = GridWorldWithTraps(50, 50, gamma, seed=seed, ergodic=True)
    elif name == "gridworld_large_sparse":
        env = GridWorldWithTraps(50, 50, gamma, seed=seed, ergodic=True, n_origins=1)
    elif name == "gridworld_hill_small":
        env = GridWorldWithTrapsAndHills(20, 20, gamma, seed=seed, ergodic=True)
    elif name == "gridworld_hill_large":
        env = GridWorldWithTrapsAndHills(50, 50, gamma, seed=seed, ergodic=True)
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

