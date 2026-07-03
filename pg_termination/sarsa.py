""" SARSA """
import time
import os
import warnings
from enum import IntEnum
import multiprocessing as mp
import warnings

import numpy as np

from pg_termination import pmd
from pg_termination import mdpmodel 
from pg_termination import utils
from pg_termination.logger import BasicLogger

TOL = 1e-10
MSG_1 = "We changed 'pi_threshold'->'pi_threshold_mult'.Please update the yaml file (defaulting to 'pi_threshold_mult=1')"

def get_loggers(settings):
    seed = settings['seed']
    validation_gamma = settings['gamma'] if (not settings['no_validation_gamma']) else 1.0
    env = mdpmodel.get_env(settings['env_name'], settings['gamma'], seed, validation_gamma)
    logger = BasicLogger(
        fname=os.path.join(settings["log_folder"], "seed=%d.csv" % seed), 
        keys=["iter", "point value", "point opt_lb", "point uni_opt_lb", 
              "agg value", "agg opt_lb", "agg uni_opt_lb", "true value", 
              "true opt_lb", "true uni_opt_lb"], 
        dtypes=['d'] + ['f'] * 9
    )
    logger_validation = BasicLogger(
        fname=os.path.join(settings["log_folder"], "validation_seed=%d.csv" % seed), 
        keys=["value", "opt_lb", "uni_opt_lb", "true value", "true opt_lb", "true uni_opt_lb"],

        dtypes=['f'] * 6,
    )
    logger_mixing = BasicLogger(
        fname=os.path.join(settings["log_folder"], "mixing_seed=%d.csv" % seed),  
        keys=["iter", "cum_time", "cum_samples", "cum_est_samples"] + ["nu_lb", "t_mix"],
        dtypes=['d', 'f', 'd', 'd'] + ['f'] * 2,
    )
    logger_ep = BasicLogger(
        fname=os.path.join(settings["log_folder"], "ep_cost_seed=%d.csv" % seed),  
        keys=["ep_cost", "ep_len", 'cum_samps'],
        dtypes=['f', 'd', 'd'],
    )

    return [logger, logger_validation, logger_mixing, logger_ep]

def _train(settings):
    logger, logger_validation, logger_mixing, logger_ep = get_loggers(settings)

    # TODO: Make this automated
    ukappa = (1.-settings['gamma'])**(-2)
    _sarsa(settings, ukappa, logger, logger_validation, logger_mixing, logger_ep)

    logger.save(max_size=10_000)
    logger_mixing.save(max_size=10_000)
    logger_validation.save(max_size=1_000)
    logger_ep.save(max_size=10_000)

def print_spmd_progress(env, t, V_t, psi_t, agg_V_t, agg_psi_t, true_V_t, true_psi_t, logger):
    """ Print and updates SPMD certificates.

    Inputs @agg_V_t and @agg_psi_t and over-written in place.

    :param env: environment from mdpmodel 
    :param t: iteration of SPMD
    :param V_t: estimated value func at iter t
    :param psi_t: estimated advantage func at iter t
    :param agg_V_t: aggregated estimate of value func at iter t
    :param agg_psi_t: aggregated estimate of advantage func at iter t
    :param true_V_t: (only TOMC, o/w arbitrary) true value func at iter t
    :param true_psi_t: (only TOMC, o/w arbitrary) true advantage func at iter t
    :param is_finite_state: (only CTD, boolean)
    :param logger: 
    """
    alpha_t = 1./(t+1)
    agg_psi_t[:,:] = (1.-alpha_t)*agg_psi_t + alpha_t*psi_t
    agg_V_t[:] = (1.-alpha_t)*agg_V_t + alpha_t*V_t
    print("Iter %d: f=%.2e (fstar_lb=%.2e) | ag_f=%.2e (ag_fstar_lb=%.2e) | true_f=%.2e (true_fstar_lb=%.2e)" % (
        t+1, 
        np.dot(env.rho, V_t), 
        np.dot(env.rho, V_t - np.max(-psi_t, axis=1)/(1.-env.gamma)), 
        np.dot(env.rho, agg_V_t), 
        np.dot(env.rho, agg_V_t - np.max(-agg_psi_t, axis=1)/(1.-env.gamma)), 
        np.dot(env.rho, true_V_t), 
        np.dot(env.rho, true_V_t - np.max(-true_psi_t, axis=1)/(1.-env.gamma)),
    ))

    logger.log(
        t+1, 
        np.dot(env.rho, V_t), 
        np.dot(env.rho, V_t - np.maximum(0, np.max(-psi_t, axis=1)/(1.-env.gamma))), 
        np.dot(env.rho, V_t - np.max(-psi_t)/(1.-env.gamma)), 
        np.dot(env.rho, agg_V_t), 
        np.dot(env.rho, agg_V_t - np.maximum(0, np.max(-agg_psi_t, axis=1)/(1.-env.gamma))), 
        np.dot(env.rho, agg_V_t - np.max(-agg_psi_t)/(1.-env.gamma)), 
        np.dot(env.rho, true_V_t), 
        np.dot(env.rho, true_V_t - np.maximum(0, np.max(-true_psi_t, axis=1)/(1.-env.gamma))),
        np.dot(env.rho, true_V_t - np.max(-true_psi_t)/(1.-env.gamma)),
    )

    return agg_V_t, agg_psi_t

def _sarsa(settings, ukappa, logger, logger_validation, logger_mixing, logger_ep, pi_0=None):
    """
    SARSA training procedure.

    :param settings: dictionary of all user-defined parameter values
    :param ukappa: estimation of ukappa (if using CTD and not set, will set warning and default to 0)
    """
    assert ukappa > 0, "ukappa=%.4e not positive" % ukappa
    seed = settings['seed']

    # initialization
    env = mdpmodel.get_env(settings['env_name'], settings['gamma'], seed)
    pi_b = np.ones((env.n_actions, env.n_states), dtype=float)/env.n_actions
    greedy_pi_t = np.ones((env.n_actions, env.n_states), dtype=float)/env.n_actions
    true_psi = true_psi_t = np.zeros((env.n_states, env.n_actions), dtype=float)
    true_V = true_V_t = np.zeros(env.n_states, dtype=float)
    cum_samples = cum_est_samples = 0
    tmix = unu = 0
    is_finite_state = env.n_states is not None
    s_time = time.time()
    e_time = -time.time()

    s_curr = env.s
    Q_t = np.zeros((env.n_states, env.n_actions), dtype=float)
    last_V_t = np.inf * np.ones(len(env.rho))
    agg_psi_t = np.zeros((env.n_states, env.n_actions), dtype=float)
    agg_V_t = np.zeros(env.n_states, dtype=float)
    ones_A = np.ones(env.n_actions)

    sa_hit_arr = np.zeros(Q_t.shape)
    alpha_t = settings['qlearn_alpha']
    has_first_log = False
    T_log_freq = -1

    a_curr = env.rng.choice(env.n_actions)
    pi_t_s = np.ones((env.n_actions, env.n_states), dtype=float)/env.n_actions

    # SPMD main for-loop
    for t in range(int(settings["n_iters"])):
        (s_next,c,term) = env.step(a_curr)

        eps_t = 1./(t+1)
        pi_t_s[:, s_next] = eps_t/env.n_actions
        pi_t_s[np.argmin(Q_t[s_next,:]), s_next] += 1-eps_t
        a_next = env.rng.choice(env.n_actions, p=pi_t_s[:,s_next])

        td_err_t = c + env.gamma*Q_t[s_next,a_next] - Q_t[s_curr,a_curr]
        if settings['qlearn_alpha'] < 0:
            # dynamic stepsize
            alpha_t = 1./(sa_hit_arr[s_curr,a_curr]+1)
            sa_hit_arr[s_curr,a_curr] += 1
        Q_t[s_curr,a_curr] = Q_t[s_curr,a_curr] + alpha_t*td_err_t
        s_curr = s_next

        V_t = np.einsum('sa,as->s', Q_t, pi_b)
        psi_t = Q_t - np.outer(V_t, ones_A) 
        
        cum_samples += 1
        if (not has_first_log) and \
            ((t == settings["n_iters"]//1000) or (e_time + time.time() >= settings["max_runtime_in_sec"]/1000)):
            has_first_log = True
            T_log_freq = max(1, t)
        # only save about 1000 logs
        if has_first_log and (t % T_log_freq == 0):
            if not settings["skip_true_model"]:
                utils.set_greedy_policy(greedy_pi_t, Q_t)
                (true_psi_t, true_V_t) = env.get_advantage(greedy_pi_t)
            print_spmd_progress(env, t, V_t, psi_t, agg_V_t, agg_psi_t, true_V_t, true_psi_t, logger)
            logger_mixing.log(t, e_time + time.time(), cum_samples, cum_est_samples, unu, tmix)
        last_V_t = V_t

        total_runtime = time.time() - s_time
        if total_runtime >= settings["max_runtime_in_sec"]:
            print("=== Breaking early because we exceeded the max runtime ===")
            break

    """
    The reason we do not do greedy is that it differs from the policy in SARSA.
    Notice in SARSA, Q_t is still going to be zero in some places, meaning we have not visited yet
    But the greedy will assign some action to that s-a pair, which will be meaningless 
    since the Q_t value therein is meaningless. So we instead use the same policy
    from SARSA instead of greedy rounding. We also saw severe degradation in performance,
    which does not reflect the on-training of SARSA
    """

    # utils.set_greedy_policy(greedy_pi_t, Q_t)
    # eps_t = 1./(settings["n_iters"])
    # greedy_pi_t *= (1-eps_t)
    # greedy_pi_t += eps_t/env.n_actions
    if not settings["skip_true_model"]:
        (true_psi, true_V) = env.get_advantage(pi_t_s)

    if settings["validation_mode"] == "random_reset":
        output = env.policy_validation_random_reset(pi_t_s, settings)
        (V, V_lb, uni_V_lb, true_V, true_V_lb, true_uni_V_lb) = output
        logger_validation.log(*output)
    else:
        pass

    print("Total runtime: %.2fs" % (time.time() - s_time))
    returned_f = np.dot(env.rho, true_V) if settings.get("tune_true_cost", False) else np.dot(env.rho, last_V_t)

    cost_arr, len_arr, cum_samps_arr = env.get_cum_cost_and_len_arr()
    for (ep_cost, ep_len, cum_samps) in zip(cost_arr, len_arr, cum_samps_arr):
        logger_ep.log(ep_cost, ep_len, cum_samps)

    return greedy_pi_t, returned_f

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
            if customized_settings['tune_exploration']:
                _train_with_tuning(customized_settings)
            else:
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
